import { Stack, useRouter, useSegments } from 'expo-router';
import { useEffect, useState } from 'react';
import { View, ActivityIndicator, Platform } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import type { Session } from '@supabase/supabase-js';
import {
  useFonts,
  Inter_400Regular,
  Inter_500Medium,
  Inter_600SemiBold,
  Inter_700Bold,
  Inter_800ExtraBold,
} from '@expo-google-fonts/inter';
import { Lora_600SemiBold, Lora_700Bold } from '@expo-google-fonts/lora';
import { supabase } from '../supabase';
import { ensureDailyReminder } from '../notifications';
import { colors, fonts } from '../theme';

// Routes that bounce straight to '/' once a session exists.
const AUTH_ONLY_ROUTES = ['login', 'signup', 'welcome', 'forgot-password'];
// reset-password is public but deliberately excluded from AUTH_ONLY_ROUTES:
// opening a recovery link establishes a session, and this screen needs to
// stay visible long enough to actually let the user set a new password
// rather than being bounced to '/' the instant that session appears.
const PUBLIC_ROUTES = [...AUTH_ONLY_ROUTES, 'reset-password'];
// A marketing landing page only makes sense on the website — a native app
// user already has intent, so they should land straight on the sign-in form.
const LOGGED_OUT_ENTRY = Platform.OS === 'web' ? '/welcome' : '/login';

export default function RootLayout() {
  // undefined = still checking for an existing session
  const [session, setSession] = useState<Session | null | undefined>(undefined);
  const router = useRouter();
  const segments = useSegments();
  const [fontsLoaded] = useFonts({
    [fonts.regular]: Inter_400Regular,
    [fonts.medium]: Inter_500Medium,
    [fonts.semibold]: Inter_600SemiBold,
    [fonts.bold]: Inter_700Bold,
    [fonts.extrabold]: Inter_800ExtraBold,
    [fonts.displaySemibold]: Lora_600SemiBold,
    [fonts.displayBold]: Lora_700Bold,
  });

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => setSession(data.session));
    const { data: sub } = supabase.auth.onAuthStateChange((_event, newSession) => {
      setSession(newSession);
    });
    return () => sub.subscription.unsubscribe();
  }, []);

  useEffect(() => {
    if (session === undefined) return;
    const segment = segments[0] ?? '';
    const isPublicRoute = PUBLIC_ROUTES.includes(segment);
    const isAuthOnlyRoute = AUTH_ONLY_ROUTES.includes(segment);
    if (!session && !isPublicRoute) {
      router.replace(LOGGED_OUT_ENTRY);
    } else if (session && isAuthOnlyRoute) {
      router.replace('/');
    }
  }, [session, segments, router]);

  useEffect(() => {
    // Schedule the 8 PM end-of-day nudge (no-op if already scheduled/denied).
    ensureDailyReminder(20, 0).catch(() => {});
  }, []);

  if (session === undefined || !fontsLoaded) {
    return (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.bg }}>
        <ActivityIndicator color={colors.accent} />
      </View>
    );
  }

  return (
    <>
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: colors.bg },
          headerShadowVisible: false,
          headerTintColor: colors.text,
          headerTitleStyle: { fontWeight: '700', fontFamily: fonts.displayBold },
          contentStyle: { backgroundColor: colors.bg },
        }}
      >
        <Stack.Screen name="welcome" options={{ headerShown: false }} />
        <Stack.Screen name="login" options={{ headerShown: false }} />
        <Stack.Screen name="signup" options={{ headerShown: false }} />
        <Stack.Screen name="forgot-password" options={{ headerShown: false }} />
        <Stack.Screen name="reset-password" options={{ headerShown: false }} />
        <Stack.Screen name="index" options={{ title: 'Today' }} />
        <Stack.Screen
          name="add"
          options={{ title: 'Add patient', presentation: 'modal' }}
        />
        <Stack.Screen name="summary" options={{ title: 'End-of-day check' }} />
      </Stack>
    </>
  );
}
