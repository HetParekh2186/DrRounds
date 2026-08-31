import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  useWindowDimensions,
} from 'react-native';
import { Link, useRouter } from 'expo-router';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { ScalePressable } from '../components/scale-pressable';
import { colors, fonts, radius, shadow } from '../theme';

const FEATURES = [
  {
    icon: '📋',
    title: 'One list, grouped by hospital',
    body: 'Your rounding list organized by hospital and room. Tap a name to mark them seen today.',
  },
  {
    icon: '🎤',
    title: 'Add patients by voice',
    body: 'Say "Sharma, Apollo, room 204, follow up on labs" and it fills in the whole form for you.',
  },
  {
    icon: '✅',
    title: 'An end-of-day safety net',
    body: 'One tap shows exactly who you haven’t marked seen yet, before you head out.',
  },
];

export default function Welcome() {
  const router = useRouter();
  const { width } = useWindowDimensions();
  const wide = width >= 720;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={[styles.nav, { maxWidth: wide ? 960 : '100%' }]}>
        <Text style={styles.wordmark}>Dr Rounds</Text>
        <View style={styles.navActions}>
          <Link href="/login" asChild>
            <ScalePressable style={styles.navSignIn}>
              <Text style={styles.navSignInText}>Sign in</Text>
            </ScalePressable>
          </Link>
          <Link href="/signup" asChild>
            <ScalePressable style={styles.navSignUp}>
              <Text style={styles.navSignUpText}>Sign up</Text>
            </ScalePressable>
          </Link>
        </View>
      </View>

      <Animated.View
        entering={FadeInUp.duration(500).springify()}
        style={[styles.hero, { maxWidth: wide ? 640 : '100%' }]}
      >
        <Text style={styles.eyebrow}>FOR DOCTORS DOING DAILY ROUNDS</Text>
        <Text style={[styles.headline, wide && styles.headlineWide]}>
          Hospital rounds, tracked simply.
        </Text>
        <Text style={styles.subhead}>
          Keep your patient list, add new patients by voice, and check that
          no one was missed — synced across every device you use.
        </Text>
        <View style={styles.heroActions}>
          <ScalePressable style={styles.primaryBtn} onPress={() => router.push('/signup')}>
            <Text style={styles.primaryBtnText}>Get started — it's free</Text>
          </ScalePressable>
          <Link href="/login" asChild>
            <ScalePressable style={styles.secondaryBtn}>
              <Text style={styles.secondaryBtnText}>Sign in</Text>
            </ScalePressable>
          </Link>
        </View>
      </Animated.View>

      <View style={[styles.features, wide && styles.featuresWide, { maxWidth: wide ? 960 : '100%' }]}>
        {FEATURES.map((f, i) => (
          <Animated.View
            key={f.title}
            entering={FadeInDown.duration(450).delay(120 + i * 90)}
            style={[styles.featureCard, wide && styles.featureCardWide]}
          >
            <Text style={styles.featureIcon}>{f.icon}</Text>
            <Text style={styles.featureTitle}>{f.title}</Text>
            <Text style={styles.featureBody}>{f.body}</Text>
          </Animated.View>
        ))}
      </View>

      <Text style={styles.footer}>Free to use. Your data stays tied to your account.</Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  content: { alignItems: 'center', paddingHorizontal: 20, paddingBottom: 60 },
  nav: {
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 24,
    paddingBottom: 40,
  },
  wordmark: { color: colors.text, fontSize: 20, fontFamily: fonts.displayBold },
  navActions: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  navSignIn: { paddingVertical: 10, paddingHorizontal: 14 },
  navSignInText: { color: colors.text, fontSize: 15, fontFamily: fonts.semibold },
  navSignUp: {
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
    paddingVertical: 10,
    paddingHorizontal: 18,
    ...shadow.card,
  },
  navSignUpText: { color: colors.accentText, fontSize: 15, fontFamily: fonts.bold },

  hero: { alignItems: 'center', width: '100%' },
  eyebrow: {
    color: colors.accent,
    fontSize: 12,
    fontFamily: fonts.bold,
    letterSpacing: 1.2,
    marginBottom: 14,
    textAlign: 'center',
  },
  headline: {
    color: colors.text,
    fontSize: 34,
    lineHeight: 40,
    fontFamily: fonts.displayBold,
    textAlign: 'center',
  },
  headlineWide: { fontSize: 44, lineHeight: 50 },
  subhead: {
    color: colors.muted,
    fontSize: 16,
    lineHeight: 24,
    fontFamily: fonts.regular,
    textAlign: 'center',
    marginTop: 16,
    maxWidth: 480,
  },
  heroActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginTop: 28,
    justifyContent: 'center',
  },
  primaryBtn: {
    backgroundColor: colors.accent,
    borderRadius: radius.md,
    paddingVertical: 15,
    paddingHorizontal: 24,
    ...shadow.raised,
  },
  primaryBtnText: { color: colors.accentText, fontSize: 16, fontFamily: fonts.bold },
  secondaryBtn: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: radius.md,
    paddingVertical: 15,
    paddingHorizontal: 24,
  },
  secondaryBtnText: { color: colors.text, fontSize: 16, fontFamily: fonts.semibold },

  features: {
    width: '100%',
    marginTop: 64,
    gap: 16,
  },
  featuresWide: { flexDirection: 'row' },
  featureCard: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: radius.lg,
    padding: 22,
    ...shadow.card,
  },
  featureCardWide: { flex: 1 },
  featureIcon: { fontSize: 26, marginBottom: 12 },
  featureTitle: {
    color: colors.text,
    fontSize: 16,
    fontFamily: fonts.bold,
    marginBottom: 6,
  },
  featureBody: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 21,
    fontFamily: fonts.regular,
  },

  footer: {
    color: colors.muted,
    fontSize: 13,
    fontFamily: fonts.regular,
    marginTop: 56,
    textAlign: 'center',
  },
});
