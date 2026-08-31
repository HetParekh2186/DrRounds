import { supabase } from './supabase';
import type { Patient } from './types';

/**
 * Local date as 'YYYY-MM-DD'. 'en-CA' formats in ISO order, and using the
 * device locale keeps "today" aligned with the doctor's actual day.
 */
export function todayStr(d: Date = new Date()): string {
  return d.toLocaleDateString('en-CA');
}

// Postgres sorts hospital names case-sensitively; re-sort client-side so
// e.g. "apollo" and "Apollo" group together, same as the old SQLite COLLATE
// NOCASE behavior. Stable sort preserves creation order within a hospital.
function sortByHospital(patients: Patient[]): Patient[] {
  return [...patients].sort((a, b) =>
    (a.hospital || '').localeCompare(b.hospital || '', undefined, { sensitivity: 'base' })
  );
}

export async function addPatient(data: {
  name: string;
  hospital?: string;
  room?: string;
  note?: string;
}): Promise<void> {
  const { error } = await supabase.from('patients').insert({
    name: data.name.trim(),
    hospital: (data.hospital ?? '').trim(),
    room: data.room?.trim() || null,
    note: data.note?.trim() || null,
  });
  if (error) throw error;
}

/** Everyone still on the rounding list (not yet discharged/removed). */
export async function getActivePatients(): Promise<Patient[]> {
  const { data, error } = await supabase
    .from('patients')
    .select('*')
    .eq('status', 'active')
    .order('created_at', { ascending: true });
  if (error) throw error;
  return sortByHospital(data ?? []);
}

/** Active patients NOT yet checked off today — the end-of-day safety net. */
export async function getUnseenToday(today: string = todayStr()): Promise<Patient[]> {
  const { data, error } = await supabase
    .from('patients')
    .select('*')
    .eq('status', 'active')
    .or(`seen_date.is.null,seen_date.neq.${today}`)
    .order('created_at', { ascending: true });
  if (error) throw error;
  return sortByHospital(data ?? []);
}

/** Toggle "seen today" on/off. Passing seen=false clears it. */
export async function setSeen(
  id: number,
  seen: boolean,
  today: string = todayStr()
): Promise<void> {
  const { error } = await supabase
    .from('patients')
    .update({ seen_date: seen ? today : null })
    .eq('id', id);
  if (error) throw error;
}

/** Patient is discharged / no longer needs visiting — drops off the active list. */
export async function dischargePatient(id: number): Promise<void> {
  const { error } = await supabase.from('patients').update({ status: 'done' }).eq('id', id);
  if (error) throw error;
}

export async function deletePatient(id: number): Promise<void> {
  const { error } = await supabase.from('patients').delete().eq('id', id);
  if (error) throw error;
}

/** Distinct hospital names already used — powers the quick-pick chips on the add screen. */
export async function getHospitals(): Promise<string[]> {
  const { data, error } = await supabase.from('patients').select('hospital').neq('hospital', '');
  if (error) throw error;
  const unique = [...new Set((data ?? []).map((r) => r.hospital))];
  return unique.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
}
