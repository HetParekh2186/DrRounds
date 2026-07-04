import { Stack } from 'expo-router';
import { SQLiteProvider } from 'expo-sqlite';
import { useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { DB_NAME, migrateDb } from '../db';
import { ensureDailyReminder } from '../notifications';
import { colors } from '../theme';

export default function RootLayout() {
  useEffect(() => {
    // Schedule the 8 PM end-of-day nudge (no-op if already scheduled/denied).
    ensureDailyReminder(20, 0).catch(() => {});
  }, []);

  return (
    <SQLiteProvider databaseName={DB_NAME} onInit={migrateDb}>
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: colors.bg },
          headerShadowVisible: false,
          headerTintColor: colors.text,
          headerTitleStyle: { fontWeight: '700' },
          contentStyle: { backgroundColor: colors.bg },
        }}
      >
        <Stack.Screen name="index" options={{ title: 'Today' }} />
        <Stack.Screen
          name="add"
          options={{ title: 'Add patient', presentation: 'modal' }}
        />
        <Stack.Screen name="summary" options={{ title: 'End-of-day check' }} />
      </Stack>
    </SQLiteProvider>
  );
}
