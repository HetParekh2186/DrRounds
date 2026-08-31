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
import Animated, {
  FadeInUp,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import { addPatient, getHospitals } from '../db';
import { parseQuickAddText } from '../quick-add';
import { useSpeechRecognition } from '../hooks/use-speech-recognition';
import { ScalePressable } from '../components/scale-pressable';
import { colors, fonts, radius, shadow } from '../theme';

export default function Add() {
  const router = useRouter();
  const { q } = useLocalSearchParams<{ q?: string }>();
  const [name, setName] = useState('');
  const [hospital, setHospital] = useState('');
  const [room, setRoom] = useState('');
  const [note, setNote] = useState('');
  const [quickText, setQuickText] = useState(q ?? '');
  const [recentHospitals, setRecentHospitals] = useState<string[]>([]);

  useEffect(() => { getHospitals().then(setRecentHospitals); }, []);

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

  const handleSpeechResult = (transcript: string, isFinal: boolean) => {
    setQuickText(transcript);
    if (isFinal && transcript.trim()) runQuickAdd(transcript);
  };

  const {
    isSupported: micSupported,
    isListening,
    start: startListening,
    stop: stopListening,
  } = useSpeechRecognition(handleSpeechResult);

  const toggleListening = () => {
    if (isListening) {
      stopListening();
    } else {
      setQuickText('');
      startListening();
    }
  };

  const micPulse = useSharedValue(1);
  useEffect(() => {
    micPulse.value = isListening
      ? withRepeat(withSequence(withTiming(1.18, { duration: 500 }), withTiming(1, { duration: 500 })), -1, true)
      : withTiming(1, { duration: 200 });
  }, [isListening, micPulse]);
  const micAnimatedStyle = useAnimatedStyle(() => ({ transform: [{ scale: micPulse.value }] }));

  const canSave = name.trim().length > 0;

  const save = async () => {
    if (!canSave) return;
    await addPatient({ name, hospital, room, note });
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
      <Animated.View entering={FadeInUp.duration(400)}>
        <Text style={styles.hint}>
          🎤 {micSupported ? 'Tap the mic below to dictate, or tap' : 'Tap'} the
          microphone on your keyboard to dictate any field.
        </Text>

        <Text style={styles.label}>Quick add (optional)</Text>
        <View style={styles.quickAddRow}>
          <TextInput
            style={[styles.input, styles.multiline, styles.quickAddInput]}
            value={quickText}
            onChangeText={setQuickText}
            onEndEditing={() => quickText.trim() && runQuickAdd(quickText)}
            placeholder="Sharma, Apollo, room 204, follow up on labs"
            placeholderTextColor={colors.muted}
            multiline
          />
          {micSupported && (
            <ScalePressable
              onPress={toggleListening}
              style={[styles.micBtn, isListening && styles.micBtnActive]}
            >
              <Animated.Text style={[styles.micIcon, micAnimatedStyle]}>
                {isListening ? '⏺' : '🎤'}
              </Animated.Text>
            </ScalePressable>
          )}
        </View>
        {isListening && <Text style={styles.listeningHint}>Listening…</Text>}

        <ScalePressable
          style={styles.quickAddBtn}
          onPress={() => quickText.trim() && runQuickAdd(quickText)}
        >
          <Text style={styles.quickAddBtnText}>Fill in fields below ↓</Text>
        </ScalePressable>

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

        <ScalePressable
          style={[styles.saveBtn, !canSave && styles.saveDisabled]}
          onPress={save}
          disabled={!canSave}
        >
          <Text style={styles.saveText}>Add to today’s list</Text>
        </ScalePressable>
      </Animated.View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  content: { padding: 20, paddingBottom: 60 },
  hint: {
    color: colors.text,
    backgroundColor: colors.accentSoft,
    borderRadius: radius.sm,
    padding: 12,
    fontSize: 14,
    fontFamily: fonts.regular,
    marginBottom: 20,
    overflow: 'hidden',
  },
  label: {
    color: colors.text,
    fontSize: 14,
    fontFamily: fonts.bold,
    marginBottom: 8,
    marginTop: 14,
  },
  input: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: radius.md,
    paddingHorizontal: 14,
    paddingVertical: 14,
    fontSize: 17,
    fontFamily: fonts.regular,
    color: colors.text,
    ...shadow.card,
  },
  multiline: { minHeight: 80, textAlignVertical: 'top' },
  quickAddRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 10 },
  quickAddInput: { flex: 1 },
  micBtn: {
    width: 52,
    height: 52,
    borderRadius: radius.pill,
    backgroundColor: colors.accentSoft,
    alignItems: 'center',
    justifyContent: 'center',
    ...shadow.card,
  },
  micBtnActive: { backgroundColor: colors.dangerSoft },
  micIcon: { fontSize: 22 },
  listeningHint: {
    color: colors.danger,
    fontSize: 13,
    fontFamily: fonts.semibold,
    marginTop: 6,
  },
  quickAddBtn: {
    backgroundColor: colors.accentSoft,
    borderRadius: radius.md,
    paddingVertical: 12,
    alignItems: 'center',
    marginTop: 10,
  },
  quickAddBtnText: { color: colors.accent, fontFamily: fonts.bold, fontSize: 15 },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 10 },
  chip: {
    backgroundColor: colors.accentSoft,
    borderRadius: radius.pill,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  chipText: { color: colors.accent, fontFamily: fonts.bold, fontSize: 14 },
  saveBtn: {
    backgroundColor: colors.accent,
    borderRadius: radius.md,
    paddingVertical: 17,
    alignItems: 'center',
    marginTop: 32,
    ...shadow.raised,
  },
  saveDisabled: { opacity: 0.4, shadowOpacity: 0 },
  saveText: { color: colors.accentText, fontSize: 17, fontFamily: fonts.bold },
});
