import { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { Link } from 'expo-router';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { supabase } from '../supabase';
import { ScalePressable } from '../components/scale-pressable';
import { colors, fonts, radius, shadow } from '../theme';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const canSubmit = email.trim().length > 0 && password.length > 0 && !loading;

  const signIn = async () => {
    if (!canSubmit) return;
    setLoading(true);
    setError('');
    const { error } = await supabase.auth.signInWithPassword({
      email: email.trim(),
      password,
    });
    setLoading(false);
    if (error) setError(error.message);
  };

  return (
    <View style={styles.container}>
      <Animated.View entering={FadeInDown.duration(500).springify()} style={styles.badge}>
        <Text style={styles.badgeText}>DR</Text>
      </Animated.View>

      <Animated.View entering={FadeInUp.duration(500).delay(80)}>
        <Text style={styles.title}>Dr Rounds</Text>
        <Text style={styles.subtitle}>Sign in to see your patient list</Text>
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

        <Text style={styles.label}>Password</Text>
        <TextInput
          style={styles.input}
          value={password}
          onChangeText={setPassword}
          placeholder="••••••••"
          placeholderTextColor={colors.muted}
          secureTextEntry
          autoComplete="password"
        />

        <Link href="/forgot-password" style={styles.forgotLink}>
          <Text style={styles.forgotLinkText}>Forgot password?</Text>
        </Link>

        <ScalePressable
          style={[styles.btn, !canSubmit && styles.btnDisabled]}
          onPress={signIn}
          disabled={!canSubmit}
        >
          {loading ? (
            <ActivityIndicator color={colors.accentText} />
          ) : (
            <Text style={styles.btnText}>Sign in</Text>
          )}
        </ScalePressable>

        <Link href="/signup" style={styles.link}>
          <Text style={styles.linkText}>New here? Create an account</Text>
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
    fontSize: 30,
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
  forgotLink: { marginTop: 10, alignSelf: 'flex-end' },
  forgotLinkText: { color: colors.muted, fontSize: 13, fontFamily: fonts.semibold },
  link: { marginTop: 20, alignSelf: 'center' },
  linkText: { color: colors.accent, fontSize: 15, fontFamily: fonts.semibold },
});
