import { useCallback, useState } from 'react';
import {
  View,
  Text,
  SectionList,
  Pressable,
  StyleSheet,
  Alert,
} from 'react-native';
import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import {
  getActivePatients,
  setSeen,
  dischargePatient,
  deletePatient,
  todayStr,
} from '../db';
import type { Patient } from '../types';
import { colors } from '../theme';

type Section = { title: string; data: Patient[] };

export default function Today() {
  const db = useSQLiteContext();
  const router = useRouter();
  const today = todayStr();
  const [sections, setSections] = useState<Section[]>([]);
  const [counts, setCounts] = useState({ seen: 0, total: 0 });

  const load = useCallback(async () => {
    const patients = await getActivePatients(db);
    const byHospital = new Map<string, Patient[]>();
    for (const p of patients) {
      const key = p.hospital || 'No hospital';
      if (!byHospital.has(key)) byHospital.set(key, []);
      byHospital.get(key)!.push(p);
    }
    setSections([...byHospital.entries()].map(([title, data]) => ({ title, data })));
    setCounts({
      seen: patients.filter((p) => p.seen_date === today).length,
      total: patients.length,
    });
  }, [db, today]);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  const toggle = async (p: Patient) => {
    await setSeen(db, p.id, p.seen_date !== today);
    load();
  };

  const onLongPress = (p: Patient) => {
    Alert.alert(p.name, 'Remove this patient from the list?', [
      {
        text: 'Discharge (done visiting)',
        onPress: async () => { await dischargePatient(db, p.id); load(); },
      },
      {
        text: 'Delete permanently',
        style: 'destructive',
        onPress: async () => { await deletePatient(db, p.id); load(); },
      },
      { text: 'Cancel', style: 'cancel' },
    ]);
  };

  return (
    <View style={styles.container}>
      <View style={styles.progress}>
        <Text style={styles.progressText}>
          {counts.total === 0
            ? 'No patients on your list yet'
            : `${counts.seen} of ${counts.total} seen today`}
        </Text>
      </View>

      <SectionList
        sections={sections}
        keyExtractor={(item) => String(item.id)}
        stickySectionHeadersEnabled={false}
        contentContainerStyle={styles.listContent}
        renderSectionHeader={({ section }) => (
          <Text style={styles.sectionHeader}>{section.title}</Text>
        )}
        renderItem={({ item }) => {
          const seen = item.seen_date === today;
          const sub = [item.room && `Room ${item.room}`, item.note]
            .filter(Boolean)
            .join(' · ');
          return (
            <Pressable
              onPress={() => toggle(item)}
              onLongPress={() => onLongPress(item)}
              style={({ pressed }) => [
                styles.row,
                seen && styles.rowSeen,
                pressed && styles.rowPressed,
              ]}
            >
              <View style={[styles.check, seen && styles.checkOn]}>
                {seen && <Text style={styles.checkMark}>✓</Text>}
              </View>
              <View style={{ flex: 1 }}>
                <Text style={[styles.name, seen && styles.nameSeen]}>{item.name}</Text>
                {!!sub && <Text style={styles.meta}>{sub}</Text>}
              </View>
            </Pressable>
          );
        }}
        ListEmptyComponent={
          <Text style={styles.empty}>
            Tap “Add patient” to start today’s list.{'\n'}Long-press a patient to
            discharge or remove.
          </Text>
        }
      />

      <View style={styles.footer}>
        <Pressable
          style={[styles.btn, styles.btnGhost]}
          onPress={() => router.push('/summary')}
        >
          <Text style={styles.btnGhostText}>End-of-day check</Text>
        </Pressable>
        <Pressable
          style={[styles.btn, styles.btnPrimary]}
          onPress={() => router.push('/add')}
        >
          <Text style={styles.btnPrimaryText}>＋ Add patient</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  progress: { paddingHorizontal: 20, paddingTop: 8, paddingBottom: 4 },
  progressText: { color: colors.muted, fontSize: 14, fontWeight: '600' },
  listContent: { paddingHorizontal: 16, paddingBottom: 120, flexGrow: 1 },
  sectionHeader: {
    color: colors.accent,
    fontSize: 13,
    fontWeight: '800',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginTop: 18,
    marginBottom: 8,
    marginLeft: 4,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderRadius: 14,
    paddingVertical: 16,
    paddingHorizontal: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: colors.line,
  },
  rowSeen: { backgroundColor: colors.accentSoft, borderColor: colors.accentSoft },
  rowPressed: { opacity: 0.7 },
  check: {
    width: 28,
    height: 28,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: colors.line,
    marginRight: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkOn: { backgroundColor: colors.accent, borderColor: colors.accent },
  checkMark: { color: colors.accentText, fontSize: 16, fontWeight: '900' },
  name: { color: colors.text, fontSize: 17, fontWeight: '600' },
  nameSeen: { color: colors.muted, textDecorationLine: 'line-through' },
  meta: { color: colors.muted, fontSize: 14, marginTop: 2 },
  empty: {
    color: colors.muted,
    textAlign: 'center',
    marginTop: 80,
    fontSize: 15,
    lineHeight: 22,
  },
  footer: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    flexDirection: 'row',
    gap: 12,
    padding: 16,
    paddingBottom: 32,
    backgroundColor: colors.bg,
    borderTopWidth: 1,
    borderTopColor: colors.line,
  },
  btn: { flex: 1, borderRadius: 14, paddingVertical: 16, alignItems: 'center' },
  btnPrimary: { backgroundColor: colors.accent },
  btnPrimaryText: { color: colors.accentText, fontSize: 16, fontWeight: '700' },
  btnGhost: { backgroundColor: colors.card, borderWidth: 1, borderColor: colors.line },
  btnGhostText: { color: colors.text, fontSize: 16, fontWeight: '600' },
});
