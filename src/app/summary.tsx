import { useCallback, useState } from 'react';
import { View, Text, FlatList, Pressable, StyleSheet } from 'react-native';
import { useFocusEffect } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { getUnseenToday, setSeen } from '../db';
import type { Patient } from '../types';
import { colors } from '../theme';

export default function Summary() {
  const db = useSQLiteContext();
  const [unseen, setUnseen] = useState<Patient[]>([]);
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    setUnseen(await getUnseenToday(db));
    setLoaded(true);
  }, [db]);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  if (loaded && unseen.length === 0) {
    return (
      <View style={styles.center}>
        <View style={styles.bigCheckWrap}>
          <Text style={styles.bigCheck}>✓</Text>
        </View>
        <Text style={styles.allDone}>All patients seen today</Text>
        <Text style={styles.sub}>Nothing left on your list. Nice work.</Text>
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
        renderItem={({ item }) => {
          const sub = [item.hospital, item.room && `Room ${item.room}`]
            .filter(Boolean)
            .join(' · ');
          return (
            <View style={styles.row}>
              <View style={{ flex: 1 }}>
                <Text style={styles.name}>{item.name}</Text>
                {!!sub && <Text style={styles.meta}>{sub}</Text>}
              </View>
              <Pressable
                style={styles.markBtn}
                onPress={async () => { await setSeen(db, item.id, true); load(); }}
              >
                <Text style={styles.markText}>Mark seen</Text>
              </Pressable>
            </View>
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
    fontSize: 18,
    fontWeight: '800',
    padding: 20,
    paddingBottom: 8,
  },
  listContent: { paddingHorizontal: 16, paddingBottom: 40 },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderRadius: 14,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: colors.line,
  },
  name: { color: colors.text, fontSize: 17, fontWeight: '600' },
  meta: { color: colors.muted, fontSize: 14, marginTop: 2 },
  markBtn: {
    backgroundColor: colors.accent,
    borderRadius: 10,
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  markText: { color: colors.accentText, fontWeight: '700' },
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
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  bigCheck: { color: colors.accentText, fontSize: 48, fontWeight: '900' },
  allDone: { color: colors.text, fontSize: 20, fontWeight: '800' },
  sub: { color: colors.muted, fontSize: 15, marginTop: 6 },
});
