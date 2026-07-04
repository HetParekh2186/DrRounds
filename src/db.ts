import * as SQLite from 'expo-sqlite';
import type { Patient } from './types';

export const DB_NAME = 'rounds.db';

/**
 * Local date as 'YYYY-MM-DD'. 'en-CA' formats in ISO order, and using the
 * device locale keeps "today" aligned with the doctor's actual day in India.
 */
export function todayStr(d: Date = new Date()): string {
  return d.toLocaleDateString('en-CA');
}

/** Runs once when the database is first opened (via SQLiteProvider onInit). */
export async function migrateDb(db: SQLite.SQLiteDatabase): Promise<void> {
  await db.execAsync(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS patients (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      name       TEXT    NOT NULL,
      hospital   TEXT    NOT NULL DEFAULT '',
      room       TEXT,
      note       TEXT,
      status     TEXT    NOT NULL DEFAULT 'active',
      seen_date  TEXT,
      created_at TEXT    NOT NULL
    );
  `);
}

export async function addPatient(
  db: SQLite.SQLiteDatabase,
  data: { name: string; hospital?: string; room?: string; note?: string }
): Promise<void> {
  await db.runAsync(
    `INSERT INTO patients (name, hospital, room, note, status, seen_date, created_at)
     VALUES (?, ?, ?, ?, 'active', NULL, ?)`,
    data.name.trim(),
    (data.hospital ?? '').trim(),
    data.room?.trim() || null,
    data.note?.trim() || null,
    new Date().toISOString()
  );
}

/** Everyone still on the rounding list (not yet discharged/removed). */
export async function getActivePatients(db: SQLite.SQLiteDatabase): Promise<Patient[]> {
  return db.getAllAsync<Patient>(
    `SELECT * FROM patients
     WHERE status = 'active'
     ORDER BY hospital COLLATE NOCASE, created_at`
  );
}

/** Active patients NOT yet checked off today — the end-of-day safety net. */
export async function getUnseenToday(
  db: SQLite.SQLiteDatabase,
  today: string = todayStr()
): Promise<Patient[]> {
  return db.getAllAsync<Patient>(
    `SELECT * FROM patients
     WHERE status = 'active' AND (seen_date IS NULL OR seen_date != ?)
     ORDER BY hospital COLLATE NOCASE, created_at`,
    today
  );
}

/** Toggle "seen today" on/off. Passing seen=false clears it. */
export async function setSeen(
  db: SQLite.SQLiteDatabase,
  id: number,
  seen: boolean,
  today: string = todayStr()
): Promise<void> {
  await db.runAsync(
    `UPDATE patients SET seen_date = ? WHERE id = ?`,
    seen ? today : null,
    id
  );
}

/** Patient is discharged / no longer needs visiting — drops off the active list. */
export async function dischargePatient(db: SQLite.SQLiteDatabase, id: number): Promise<void> {
  await db.runAsync(`UPDATE patients SET status = 'done' WHERE id = ?`, id);
}

export async function deletePatient(db: SQLite.SQLiteDatabase, id: number): Promise<void> {
  await db.runAsync(`DELETE FROM patients WHERE id = ?`, id);
}

/** Distinct hospital names already used — powers the quick-pick chips on the add screen. */
export async function getHospitals(db: SQLite.SQLiteDatabase): Promise<string[]> {
  const rows = await db.getAllAsync<{ hospital: string }>(
    `SELECT DISTINCT hospital FROM patients
     WHERE hospital != '' ORDER BY hospital COLLATE NOCASE`
  );
  return rows.map((r) => r.hospital);
}
