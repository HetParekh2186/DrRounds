import { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  ActivityIndicator,
  Platform,
} from 'react-native';
import { Link } from 'expo-router';
import Animated, { FadeInDown, FadeInUp, ZoomIn } from 'react-native-reanimated';
import { supabase } from '../supabase';
import { ScalePressable } from '../components/scale-pressable';
import { colors, fonts, radius, shadow } from '../theme';

export default function ForgotPassword() {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [sent, setSent] = useState(false);

  const canSubmit = email.trim().length > 0 && !loading;

  const sendResetLink = async () => {
    if (!canSubmit) return;
    setLoading(true);
    setError('');
    const redirectTo =
      Platform.OS === 'web' && typeof window !== 'undefined'
        ? `${window.location.origin}/reset-password`
        : undefined;
    const { error } = await supabase.auth.resetPasswordForEmail(email.trim(), { redirectTo });
    setLoading(false);
    if (error) {
      setError(error.message);
      return;
    }
    setSent(true);
  };

  if (sent) {
    return (
      <View style={styles.container}>
        <Animated.View entering={ZoomIn.duration(400).springify()} style={styles.badge}>
          <Text style={styles.badgeText}>✓</Text>
        </Animated.View>
        <Animated.View entering={FadeInUp.duration(400).delay(100)}>
          <Text style={styles.title}>Check your email</Text>
          <Text style={styles.subtitle}>
            If an account exists for {email.trim()}, we sent a link to reset your password.
          </Text>
          <Link href="/login" style={styles.link}>
            <Text style={styles.linkText}>Back to sign in</Text>
          </Link>
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
        <Text style={styles.title}>Reset your password</Text>
        <Text style={styles.subtitle}>We'll email you a link to set a new one</Text>
      </Animated.View>

      {!!error && (
        <Animated.Text entering={FadeInDown.duration(250)} style={styles.error}>
          {error}
        </Animated.Text>
      )}

      <Animated.View entering={FadeInUp.duration(500).delay(140)}>
        <Text style={styles.label}>Email</Text>
        <TextInput
          style={styles.input}
          value={email}
          onChangeText={setEmail}
          placeholder="you@example.com"
          placeholderTextColor={colors.muted}
          autoCapitalize="none"
          autoComplete="email"
          keyboardType="email-address"
        />

        <ScalePressable
          style={[styles.btn, !canSubmit && styles.btnDisabled]}
          onPress={sendResetLink}
          disabled={!canSubmit}
        >
          {loading ? (
            <ActivityIndicator color={colors.accentText} />
          ) : (
            <Text style={styles.btnText}>Send reset link</Text>
          )}
        </ScalePressable>

        <Link href="/login" style={styles.link}>
          <Text style={styles.linkText}>Back to sign in</Text>
        </Link>
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
