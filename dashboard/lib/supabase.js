// Supabase REST client for the static dashboard.
//
// Loaded via ESM CDN so there's no build step — any page just imports from here.
// The anon key is safe to publish: RLS policies (infra/rls.sql) restrict it to
// SELECT-only.

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const SUPABASE_URL  = 'https://zbqujavkizijhiveqoxv.supabase.co';
const SUPABASE_ANON = 'sb_publishable_HoTpPorRxnJbS0vTWVmyWw_TMTUUUrI';

export const sb = createClient(SUPABASE_URL, SUPABASE_ANON);

/** Public URL for an artifact path like "models/abc/weights.pt".
 * `models`, `logs`, and `replays` are public — dashboard reads don't need
 * signing. */
export function publicUrl(path) {
  if (!path) return null;
  return `${SUPABASE_URL}/storage/v1/object/public/${path}`;
}

// Back-compat export: old callers import `signedUrl`. Public URL works the
// same for the dashboard's purposes; kept sync to simplify call sites.
export function signedUrl(path) {
  return publicUrl(path);
}

/** Read the current URL's ?id=... (for per-entity detail pages). */
export function queryParam(name) {
  return new URLSearchParams(window.location.search).get(name);
}
