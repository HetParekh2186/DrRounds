import { useCallback, useState } from 'react';
import { View, Text, FlatList, StyleSheet } from 'react-native';
import { useFocusEffect } from 'expo-router';
import Animated, { FadeInDown, FadeInUp, ZoomIn } from 'react-native-reanimated';
import { getUnseenToday, setSeen } from '../db';
import { ScalePressable } from '../components/scale-pressable';
import type { Patient } from '../types';
import { colors, fonts, radius, shadow } from '../theme';

export default function Summary() {
  const [unseen, setUnseen] = useState<Patient[]>([]);
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    setUnseen(await getUnseenToday());
    setLoaded(true);
  }, []);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  if (loaded && unseen.length === 0) {
    return (
      <View style={styles.center}>
        <Animated.View entering={ZoomIn.duration(450).springify()} style={styles.bigCheckWrap}>
          <Text style={styles.bigCheck}>✓</Text>
        </Animated.View>
        <Animated.View entering={FadeInUp.duration(400).delay(120)}>
          <Text style={styles.allDone}>All patients seen today</Text>
          <Text style={styles.sub}>Nothing left on your list. Nice work.</Text>
        </Animated.View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {unseen.length > 0 && (
        <Text style={styles.heading}>
          {unseen.length} patient{unseen.length > 1 ? 's' : ''} still to visit
        </Text>
      )}
      <FlatList
        data={unseen}
        keyExtractor={(i) => String(i.id)}
        contentContainerStyle={styles.listContent}
        renderItem={({ item, index }) => {
          const sub = [item.hospital, item.room && `Room ${item.room}`]
            .filter(Boolean)
            .join(' · ');
          return (
            <Animated.View
              entering={FadeInDown.duration(300).delay(Math.min(index, 8) * 40)}
              style={styles.row}
            >
              <View style={{ flex: 1 }}>
                <Text style={styles.name}>{item.name}</Text>
                {!!sub && <Text style={styles.meta}>{sub}</Text>}
              </View>
              <ScalePressable
                style={styles.markBtn}
                onPress={async () => { await setSeen(item.id, true); load(); }}
              >
                <Text style={styles.markText}>Mark seen</Text>
              </ScalePressable>
            </Animated.View>
          );
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  heading: {
    color: colors.text,
    fontSize: 19,
    fontFamily: fonts.displayBold,
    padding: 20,
    paddingBottom: 8,
  },
  listContent: { paddingHorizontal: 16, paddingBottom: 40 },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderRadius: radius.md,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: colors.line,
    ...shadow.card,
  },
  name: { color: colors.text, fontSize: 17, fontFamily: fonts.semibold },
  meta: { color: colors.muted, fontSize: 14, fontFamily: fonts.regular, marginTop: 2 },
  markBtn: {
    backgroundColor: colors.success,
    borderRadius: radius.sm,
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  markText: { color: colors.accentText, fontFamily: fonts.bold },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.bg,
    padding: 40,
  },
  bigCheckWrap: {
    width: 88,
    height: 88,
    borderRadius: 44,
    backgroundColor: colors.success,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    ...shadow.raised,
  },
  bigCheck: { color: colors.accentText, fontSize: 48, fontFamily: fonts.extrabold },
  allDone: { color: colors.text, fontSize: 21, fontFamily: fonts.displayBold, textAlign: 'center' },
  sub: { color: colors.muted, fontSize: 15, fontFamily: fonts.regular, marginTop: 6, textAlign: 'center' },
});
