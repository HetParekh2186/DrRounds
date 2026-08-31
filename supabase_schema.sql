create table patients (
  id bigint generated always as identity primary key,
  user_id uuid not null references auth.users(id) on delete cascade default auth.uid(),
  name text not null,
  hospital text not null default '',
  room text,
  note text,
  status text not null default 'active',
  seen_date date,
  created_at timestamptz not null default now()
);

create index patients_user_status_idx on patients (user_id, status);

alter table patients enable row level security;

create policy "select_own_patients" on patients
  for select using (auth.uid() = user_id);

create policy "insert_own_patients" on patients
  for insert with check (auth.uid() = user_id);

create policy "update_own_patients" on patients
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "delete_own_patients" on patients
  for delete using (auth.uid() = user_id);

-- "Automatically expose new tables" is off, so grant access explicitly.
-- Only the authenticated role gets access — anon has none, RLS aside.
grant select, insert, update, delete on patients to authenticated;
