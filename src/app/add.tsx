import { useEffect, useState } from 'react';
import {
  View,
  Text,
  TextInput,
  Pressable,
  StyleSheet,
  ScrollView,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { addPatient, getHospitals } from '../db';
import { parseQuickAddText } from '../quick-add';
import { colors } from '../theme';

export default function Add() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { q } = useLocalSearchParams<{ q?: string }>();
  const [name, setName] = useState('');
  const [hospital, setHospital] = useState('');
  const [room, setRoom] = useState('');
  const [note, setNote] = useState('');
  const [quickText, setQuickText] = useState(q ?? '');
  const [recentHospitals, setRecentHospitals] = useState<string[]>([]);

  useEffect(() => { getHospitals(db).then(setRecentHospitals); }, [db]);

  // Prefills from a Shortcut deep link like /add?q=Sharma,+Apollo,+room+204 —
  // parsed as soon as we know which hospitals are already recognized.
  useEffect(() => {
    if (q) runQuickAdd(q);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, recentHospitals]);

  const runQuickAdd = (text: string) => {
    const parsed = parseQuickAddText(text, recentHospitals);
    setName(parsed.name);
    if (parsed.hospital) setHospital(parsed.hospital);
    if (parsed.room) setRoom(parsed.room);
    if (parsed.note) setNote(parsed.note);
  };

  const canSave = name.trim().length > 0;

  const save = async () => {
    if (!canSave) return;
    await addPatient(db, { name, hospital, room, note });
    // A Shortcut deep link opens this screen directly, with no in-app
    // history to go back to — router.back() would silently no-op then.
    if (router.canGoBack()) {
      router.back();
    } else {
      router.replace('/');
    }
  };

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      keyboardShouldPersistTaps="handled"
    >
      <Text style={styles.hint}>
        🎤 Tap the microphone on your keyboard to dictate any field, or use
        Quick add below to fill everything from one sentence.
      </Text>

      <Text style={styles.label}>Quick add (optional)</Text>
      <TextInput
        style={[styles.input, styles.multiline]}
        value={quickText}
        onChangeText={setQuickText}
        onEndEditing={() => quickText.trim() && runQuickAdd(quickText)}
        placeholder="Sharma, Apollo, room 204, follow up on labs"
        placeholderTextColor={colors.muted}
        multiline
      />
      <Pressable
        style={styles.quickAddBtn}
        onPress={() => quickText.trim() && runQuickAdd(quickText)}
      >
        <Text style={styles.quickAddBtnText}>Fill in fields below ↓</Text>
      </Pressable>

      <Text style={styles.label}>Patient (name, initials, or a code)</Text>
      <TextInput
        style={styles.input}
        value={name}
        onChangeText={setName}
        placeholder="e.g. Mr. Sharma  /  R.S.  /  Bed 12"
        placeholderTextColor={colors.muted}
        autoFocus
      />

      <Text style={styles.label}>Hospital</Text>
      <TextInput
        style={styles.input}
        value={hospital}
        onChangeText={setHospital}
        placeholder="e.g. Apollo"
        placeholderTextColor={colors.muted}
      />
      {recentHospitals.length > 0 && (
        <View style={styles.chips}>
          {recentHospitals.map((h) => (
            <Pressable key={h} style={styles.chip} onPress={() => setHospital(h)}>
              <Text style={styles.chipText}>{h}</Text>
            </Pressable>
          ))}
        </View>
      )}

      <Text style={styles.label}>Room / bed (optional)</Text>
      <TextInput
        style={styles.input}
        value={room}
        onChangeText={setRoom}
        placeholder="e.g. 204"
        placeholderTextColor={colors.muted}
      />

      <Text style={styles.label}>Note (optional)</Text>
      <TextInput
        style={[styles.input, styles.multiline]}
        value={note}
        onChangeText={setNote}
        placeholder="e.g. follow-up, review reports"
        placeholderTextColor={colors.muted}
        multiline
      />

      <Pressable
        style={[styles.saveBtn, !canSave && styles.saveDisabled]}
        onPress={save}
        disabled={!canSave}
      >
        <Text style={styles.saveText}>Add to today’s list</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  content: { padding: 20, paddingBottom: 60 },
  hint: {
    color: colors.text,
    backgroundColor: colors.accentSoft,
    borderRadius: 10,
    padding: 12,
    fontSize: 14,
    marginBottom: 20,
    overflow: 'hidden',
  },
  label: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '700',
    marginBottom: 8,
    marginTop: 14,
  },
  input: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 14,
    fontSize: 17,
    color: colors.text,
  },
  multiline: { minHeight: 80, textAlignVertical: 'top' },
  quickAddBtn: {
    backgroundColor: colors.accentSoft,
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: 'center',
    marginTop: 10,
  },
  quickAddBtnText: { color: colors.accent, fontWeight: '700', fontSize: 15 },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 10 },
  chip: {
    backgroundColor: colors.accentSoft,
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  chipText: { color: colors.accent, fontWeight: '700', fontSize: 14 },
  saveBtn: {
    backgroundColor: colors.accent,
    borderRadius: 14,
    paddingVertical: 17,
    alignItems: 'center',
    marginTop: 32,
  },
  saveDisabled: { opacity: 0.4 },
  saveText: { color: colors.accentText, fontSize: 17, fontWeight: '700' },
});
