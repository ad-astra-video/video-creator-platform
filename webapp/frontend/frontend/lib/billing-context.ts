/**
 * Billing attribution context.
 *
 * The Worker records per-project spend in its durable ledger keyed by an
 * `X-VC-Project-Id` header on POST /sign-ticket. This tiny leaf module (no
 * imports, so it can never create a dependency cycle) tracks the *currently
 * active* project so the media/sign path can stamp that header without every
 * caller threading the id down through the transport stack.
 *
 * GenSpace sets the active project when it mounts (alongside
 * setActiveGenerationOwner in generation-recovery.ts); the sign path reads it
 * back at ticket time. When unset (no project mounted / older flow), spend is
 * simply left unattributed — it still counts toward the global total.
 */

let activeProjectId: string | null = null

export function setBillingProjectId(projectId: string | null): void {
  activeProjectId = projectId
}

export function getBillingProjectId(): string | null {
  return activeProjectId
}
