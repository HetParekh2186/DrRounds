import { useEffect, useState } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { Link, useRouter } from 'expo-router';
import Animated, { FadeInDown, FadeInUp, ZoomIn } from 'react-native-reanimated';
import { supabase } from '../supabase';
import { ScalePressable } from '../components/scale-pressable';
import { colors, fonts, radius, shadow } from '../theme';

export default function ResetPassword() {
  const router = useRouter();
  // undefined = still checking, true = came from a valid recovery link (or
  // is otherwise signed in), false = no valid session to reset a password for
  const [ready, setReady] = useState<boolean | undefined>(undefined);
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [done, setDone] = useState(false);

  useEffect(() => {
    // The recovery link's token is parsed into a session automatically
    // (detectSessionInUrl), which fires a PASSWORD_RECOVERY event — but if a
    // session already exists by the time this mounts, check for that too.
    supabase.auth.getSession().then(({ data }) => {
      if (data.session) setReady(true);
    });
    const { data: sub } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === 'PASSWORD_RECOVERY' || session) setReady(true);
    });
    const timeout = setTimeout(() => setReady((r) => r ?? false), 3000);
    return () => {
      sub.subscription.unsubscribe();
      clearTimeout(timeout);
    };
  }, []);

  const canSubmit = password.length >= 6 && !loading;

  const updatePassword = async () => {
    if (!canSubmit) return;
    setLoading(true);
    setError('');
    const { error } = await supabase.auth.updateUser({ password });
    setLoading(false);
    if (error) {
      setError(error.message);
      return;
    }
    setDone(true);
    setTimeout(() => router.replace('/'), 1500);
  };

  if (ready === undefined) {
    return (
      <View style={styles.container}>
        <ActivityIndicator color={colors.accent} />
      </View>
    );
  }

  if (ready === false) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>Link expired</Text>
        <Text style={styles.subtitle}>
          This password reset link is invalid or has expired.
        </Text>
        <Link href="/forgot-password" style={styles.link}>
          <Text style={styles.linkText}>Request a new link</Text>
        </Link>
      </View>
    );
  }

  if (done) {
    return (
      <View style={styles.container}>
        <Animated.View entering={ZoomIn.duration(400).springify()} style={styles.badge}>
          <Text style={styles.badgeText}>✓</Text>
        </Animated.View>
        <Animated.View entering={FadeInUp.duration(400).delay(100)}>
          <Text style={styles.title}>Password updated</Text>
          <Text style={styles.subtitle}>Taking you to your patient list…</Text>
        </Animated.View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Animated.View entering={FadeInDown.duration(500).springify()} style={styles.badge}>
        <Text style={styles.badgeText}>DR</Text>
      </Animated.View>

      <Animated.View entering={FadeInUp.duration(500).delay(80)}>
        <Text style={styles.title}>Set a new password</Text>
        <Text style={styles.subtitle}>Choose something you haven't used before</Text>
      </Animated.View>

      {!!error && (
        <Animated.Text entering={FadeInDown.duration(250)} style={styles.error}>
          {error}
        </Animated.Text>
      )}

      <Animated.View entering={FadeInUp.duration(500).delay(140)}>
        <Text style={styles.label}>New password (min 6 characters)</Text>
        <TextInput
          style={styles.input}
          value={password}
          onChangeText={setPassword}
          placeholder="••••••••"
          placeholderTextColor={colors.muted}
          secureTextEntry
          autoComplete="new-password"
        />

        <ScalePressable
          style={[styles.btn, !canSubmit && styles.btnDisabled]}
          onPress={updatePassword}
          disabled={!canSubmit}
        >
          {loading ? (
            <ActivityIndicator color={colors.accentText} />
          ) : (
            <Text style={styles.btnText}>Update password</Text>
          )}
        </ScalePressable>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg, padding: 24, justifyContent: 'center' },
  badge: {
    width: 64,
    height: 64,
    borderRadius: radius.lg,
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'center',
    marginBottom: 20,
    ...shadow.raised,
  },
  badgeText: { color: colors.accentText, fontSize: 22, fontFamily: fonts.displayBold },
  title: {
    color: colors.text,
    fontSize: 26,
    fontFamily: fonts.displayBold,
    textAlign: 'center',
  },
  subtitle: {
    color: colors.muted,
    fontSize: 15,
    fontFamily: fonts.regular,
    textAlign: 'center',
    marginTop: 6,
    marginBottom: 32,
  },
  error: {
    color: colors.danger,
    backgroundColor: colors.dangerSoft,
    borderRadius: radius.sm,
    padding: 12,
    fontSize: 14,
    fontFamily: fonts.medium,
    marginBottom: 16,
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
  btn: {
    backgroundColor: colors.accent,
    borderRadius: radius.md,
    paddingVertical: 17,
    alignItems: 'center',
    marginTop: 28,
    ...shadow.raised,
  },
  btnDisabled: { opacity: 0.4, shadowOpacity: 0 },
  btnText: { color: colors.accentText, fontSize: 16, fontFamily: fonts.bold },
  link: { marginTop: 20, alignSelf: 'center' },
  linkText: { color: colors.accent, fontSize: 15, fontFamily: fonts.semibold },
});
