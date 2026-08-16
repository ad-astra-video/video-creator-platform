/**
 * Optimistic-balance resolution + reconciliation.
 *
 * PymtHouse debits a user's allowance ASYNCHRONOUSLY (its metering pipeline
 * consumes after the orchestrator redeems a signed ticket), so reading
 * `/users/{id}/allowances` directly lags ticket-send time. To show the spend
 * the moment a ticket is signed, we subtract the OUTSTANDING optimistic-debit
 * mirror (see ledger.ts + migrations/0003) from the authoritative balance when
 * displaying it, and reconcile the mirror away as PymtHouse catches up.
 *
 * Reconciliation guarantees (no double counting):
 *   - authority  = PymtHouse remaining entitlement (source of truth)
 *   - pending    = mirrored ticket expected values (ticketEV) not yet absorbed
 *   - displayed  = max(0, authority − pending)
 *   - A mirrored row is ABSORBED (deleted) oldest-first as far as PymtHouse's
 *     `consumedUsdMicros` has advanced since our per-user baseline — once
 *     PymtHouse has consumed it, the authoritative balance already reflects the
 *     drop, so we must stop subtracting it.
 *   - A row older than OPTIMISTIC_DEBIT_TTL is pruned (safety net): if the ticket
 *     was never redeemed, the money wasn't spent and the display should recover
 *     it; if it WAS redeemed, absorbed via the consumed-delta above.
 *   - The baseline (`balance_sync.last_consumed`) is lazily seeded on a user's
 *     first recorded debit so we never mis-attribute pre-existing consumption
 *     to our own tickets.
 *
 * This NEVER writes the authoritative balance — it only reads it and mirrors
 * what we signed. Money moves exclusively through PymtHouse.
 */

import { PymtHouseClient } from "./pymthouse";
import {
  getBalanceSync,
  listOptimisticDebits,
  deleteOptimisticDebit,
  pruneOptimisticDebits,
  setBalanceSync,
  sumOptimisticDebits,
} from "./ledger";
import type { Balance, Env } from "./types";

/** Mirrored debits older than this are pruned as "not actually spent" (unredeemed). */
export const OPTIMISTIC_DEBIT_TTL_MS = 24 * 60 * 60 * 1000;

export interface UserBalanceView extends Balance {
  /** Outstanding optimistic debits not yet absorbed by PymtHouse metering. */
  pendingUsdMicros: string;
  /** Raw authoritative PymtHouse remaining balance (before optimistic subtraction). */
  authorityUsdMicros: string;
}

/**
 * Fetch the authoritative PymtHouse balance, reconcile the optimistic-debit
 * mirror against PymtHouse `consumed`, and return the net view to display.
 * Throws when PymtHouse is unreachable (callers decide how to degrade).
 */
export async function resolveUserBalance(env: Env, externalUserId: string): Promise<UserBalanceView> {
  const client = new PymtHouseClient(env);
  const authority: Balance = await client.getBalance(externalUserId);

  // 1) Prune mirrored debits older than the TTL (unredeemed-ticket safety net).
  const cutoff = new Date(Date.now() - OPTIMISTIC_DEBIT_TTL_MS).toISOString().replace("T", " ").slice(0, 19);
  await pruneOptimisticDebits(env.DB, externalUserId, cutoff);

  // 2) Absorb mirrored rows as far as PymtHouse's consumed has advanced since our baseline.
  const consumed = BigInt(authority.consumedUsdMicros || "0");
  const sync = await getBalanceSync(env.DB, externalUserId);
  if (sync) {
    let delta = consumed - BigInt(sync.last_consumed_usd_micros || 0);
    if (delta > 0n) {
      const rows = await listOptimisticDebits(env.DB, externalUserId);
      for (const row of rows) {
        if (delta <= 0n) break;
        const face = BigInt(row.expected_value_usd_micros || 0);
        if (delta >= face) {
          await deleteOptimisticDebit(env.DB, row.id);
          delta -= face;
        } else {
          // Partially consumed: leave the row; self-corrects on the next read.
          delta = 0n;
        }
      }
    }
    await setBalanceSync(env.DB, externalUserId, consumed);
  }
  // No baseline yet => nothing recorded for this user yet, or first sight; skip.
  // (If debits exist but baseline was somehow lost, they're caught by the TTL.)

  // 3) Net displayed balance = authority − outstanding pending.
  const pending = await sumOptimisticDebits(env.DB, externalUserId, OPTIMISTIC_DEBIT_TTL_MS);
  const authorityBalance = BigInt(authority.balanceUsdMicros || "0");
  const net = authorityBalance >= pending ? authorityBalance - pending : 0n;
  const netStr = net.toString();

  return {
    hasAccess: authority.hasAccess,
    balanceUsdMicros: netStr,
    remainingUsdMicros: netStr,
    consumedUsdMicros: authority.consumedUsdMicros,
    lifetimeGrantedUsdMicros: authority.lifetimeGrantedUsdMicros,
    pendingUsdMicros: pending.toString(),
    authorityUsdMicros: authority.balanceUsdMicros,
  };
}
