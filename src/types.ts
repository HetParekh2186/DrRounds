export type PatientStatus = 'active' | 'done';

export interface Patient {
  id: number;
  name: string;
  hospital: string;
  room: string | null;
  note: string | null;
  status: PatientStatus;
  seen_date: string | null; // 'YYYY-MM-DD' of the day the patient was last checked off
  created_at: string; // ISO timestamp
}
