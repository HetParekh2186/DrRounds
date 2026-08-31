import { useCallback, useEffect, useState } from 'react';
import {
  View,
  Text,
  SectionList,
  Pressable,
  StyleSheet,
  Modal,
} from 'react-native';
import { Stack, useFocusEffect, useRouter } from 'expo-router';
import Animated, {
  SlideInDown,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import {
  getActivePatients,
  setSeen,
  dischargePatient,
  deletePatient,
  todayStr,
} from '../db';
import { supabase } from '../supabase';
import { ScalePressable } from '../components/scale-pressable';
import type { Patient } from '../types';
import { colors, fonts, radius, shadow } from '../theme';

type Section = { title: string; data: Patient[] };

function ProgressBar({ seen, total }: { seen: number; total: number }) {
  const pct = useSharedValue(0);

  useEffect(() => {
    pct.value = withTiming(total === 0 ? 0 : seen / total, { duration: 450 });
  }, [seen, total, pct]);

  const fillStyle = useAnimatedStyle(() => ({
    width: `${pct.value * 100}%`,
  }));

  return (
    <View style={styles.progressTrack}>
      <Animated.View style={[styles.progressFill, fillStyle]} />
    </View>
  );
}

function PatientRow({
  item,
  today,
  onToggle,
  onLongPress,
}: {
  item: Patient;
  today: string;
  onToggle: (p: Patient) => void;
  onLongPress: (p: Patient) => void;
}) {
  const seen = item.seen_date === today;
  const checkScale = useSharedValue(seen ? 1 : 0.6);

  useEffect(() => {
    checkScale.value = withSpring(seen ? 1 : 0.6, { damping: 12, stiffness: 320 });
  }, [seen, checkScale]);

  const checkAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: checkScale.value }],
  }));

  const sub = [item.room && `Room ${item.room}`, item.note].filter(Boolean).join(' · ');

  return (
    <ScalePressable
      onPress={() => onToggle(item)}
      onLongPress={() => onLongPress(item)}
      style={[styles.row, seen && styles.rowSeen]}
      scaleTo={0.98}
    >
      <View style={[styles.check, seen && styles.checkOn]}>
        {seen && (
          <Animated.Text style={[styles.checkMark, checkAnimatedStyle]}>✓</Animated.Text>
        )}
      </View>
      <View style={{ flex: 1 }}>
        <Text style={[styles.name, seen && styles.nameSeen]}>{item.name}</Text>
        {!!sub && <Text style={styles.meta}>{sub}</Text>}
      </View>
    </ScalePressable>
  );
}

export default function Today() {
  const router = useRouter();
  const today = todayStr();
  const [sections, setSections] = useState<Section[]>([]);
  const [counts, setCounts] = useState({ seen: 0, total: 0 });
  const [actionTarget, setActionTarget] = useState<Patient | null>(null);

  const load = useCallback(async () => {
    const patients = await getActivePatients();
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
  }, [today]);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  const toggle = async (p: Patient) => {
    await setSeen(p.id, p.seen_date !== today);
    load();
  };

  const onLongPress = (p: Patient) => setActionTarget(p);

  const handleDischarge = async () => {
    if (!actionTarget) return;
    await dischargePatient(actionTarget.id);
    setActionTarget(null);
    load();
  };

  const handleDelete = async () => {
    if (!actionTarget) return;
    await deletePatient(actionTarget.id);
    setActionTarget(null);
    load();
  };

  const signOut = () => {
    supabase.auth.signOut();
  };

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          headerRight: () => (
            <Pressable onPress={signOut} hitSlop={10}>
              <Text style={styles.signOut}>Sign out</Text>
            </Pressable>
          ),
        }}
      />
      <View style={styles.progress}>
        <Text style={styles.progressText}>
          {counts.total === 0
            ? 'No patients on your list yet'
            : `${counts.seen} of ${counts.total} seen today`}
        </Text>
        {counts.total > 0 && <ProgressBar seen={counts.seen} total={counts.total} />}
      </View>

      <SectionList
        sections={sections}
        keyExtractor={(item) => String(item.id)}
        stickySectionHeadersEnabled={false}
        contentContainerStyle={styles.listContent}
        renderSectionHeader={({ section }) => (
          <Text style={styles.sectionHeader}>{section.title}</Text>
        )}
        renderItem={({ item }) => (
          <PatientRow item={item} today={today} onToggle={toggle} onLongPress={onLongPress} />
        )}
        ListEmptyComponent={
          <Text style={styles.empty}>
            Tap “Add patient” to start today’s list.{'\n'}Long-press a patient to
            discharge or remove.
          </Text>
        }
      />

      <View style={styles.footer}>
        <ScalePressable
          style={[styles.btn, styles.btnGhost]}
          onPress={() => router.push('/summary')}
        >
          <Text style={styles.btnGhostText}>End-of-day check</Text>
        </ScalePressable>
        <ScalePressable
          style={[styles.btn, styles.btnPrimary]}
          onPress={() => router.push('/add')}
        >
          <Text style={styles.btnPrimaryText}>＋ Add patient</Text>
        </ScalePressable>
      </View>

      <Modal
        visible={!!actionTarget}
        transparent
        animationType="fade"
        onRequestClose={() => setActionTarget(null)}
      >
        <Pressable style={styles.sheetBackdrop} onPress={() => setActionTarget(null)}>
          <Animated.View
            entering={SlideInDown.duration(280).springify().damping(18)}
            style={styles.sheetCard}
          >
            <Pressable onPress={() => {}}>
              <Text style={styles.sheetTitle}>{actionTarget?.name}</Text>
              <Text style={styles.sheetSubtitle}>Remove this patient from the list?</Text>

              <Pressable style={styles.sheetBtn} onPress={handleDischarge}>
                <Text style={styles.sheetBtnText}>Discharge (done visiting)</Text>
              </Pressable>
              <Pressable style={styles.sheetBtn} onPress={handleDelete}>
                <Text style={styles.sheetBtnDanger}>Delete permanently</Text>
              </Pressable>
              <Pressable style={styles.sheetBtn} onPress={() => setActionTarget(null)}>
                <Text style={styles.sheetBtnText}>Cancel</Text>
              </Pressable>
            </Pressable>
          </Animated.View>
        </Pressable>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  signOut: { color: colors.accent, fontSize: 15, fontFamily: fonts.semibold, marginRight: 4 },
  progress: { paddingHorizontal: 20, paddingTop: 8, paddingBottom: 10 },
  progressText: { color: colors.muted, fontSize: 14, fontFamily: fonts.semibold },
  progressTrack: {
    height: 6,
    borderRadius: radius.pill,
    backgroundColor: colors.line,
    marginTop: 8,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: radius.pill,
    backgroundColor: colors.accent,
  },
  listContent: { paddingHorizontal: 16, paddingBottom: 120, flexGrow: 1 },
  sectionHeader: {
    color: colors.accent,
    fontSize: 13,
    fontFamily: fonts.bold,
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
    borderRadius: radius.lg,
    paddingVertical: 16,
    paddingHorizontal: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: colors.line,
    ...shadow.card,
  },
  rowSeen: { backgroundColor: colors.successSoft, borderColor: colors.successSoft },
  check: {
    width: 26,
    height: 26,
    borderRadius: radius.pill,
    borderWidth: 2,
    borderColor: colors.line,
    marginRight: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkOn: { backgroundColor: colors.success, borderColor: colors.success },
  checkMark: { color: colors.accentText, fontSize: 16, fontFamily: fonts.extrabold },
  name: { color: colors.text, fontSize: 17, fontFamily: fonts.semibold },
  nameSeen: { color: colors.muted, textDecorationLine: 'line-through' },
  meta: { color: colors.muted, fontSize: 14, fontFamily: fonts.regular, marginTop: 2 },
  empty: {
    color: colors.muted,
    textAlign: 'center',
    marginTop: 80,
    fontSize: 15,
    fontFamily: fonts.regular,
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
  btn: { flex: 1, borderRadius: radius.md, paddingVertical: 16, alignItems: 'center' },
  btnPrimary: { backgroundColor: colors.accent, ...shadow.raised },
  btnPrimaryText: { color: colors.accentText, fontSize: 16, fontFamily: fonts.bold },
  btnGhost: { backgroundColor: colors.card, borderWidth: 1, borderColor: colors.line },
  btnGhostText: { color: colors.text, fontSize: 16, fontFamily: fonts.semibold },
  sheetBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-end',
  },
  sheetCard: {
    backgroundColor: colors.card,
    borderTopLeftRadius: radius.lg,
    borderTopRightRadius: radius.lg,
    padding: 20,
    paddingBottom: 36,
    ...shadow.raised,
  },
  sheetTitle: { color: colors.text, fontSize: 19, fontFamily: fonts.displayBold, textAlign: 'center' },
  sheetSubtitle: {
    color: colors.muted,
    fontSize: 14,
    fontFamily: fonts.regular,
    textAlign: 'center',
    marginTop: 4,
    marginBottom: 16,
  },
  sheetBtn: {
    paddingVertical: 16,
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: colors.line,
  },
  sheetBtnText: { color: colors.text, fontSize: 16, fontFamily: fonts.semibold },
  sheetBtnDanger: { color: colors.danger, fontSize: 16, fontFamily: fonts.semibold },
});
