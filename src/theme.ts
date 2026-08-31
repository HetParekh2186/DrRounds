// "Sunroom" — warm sand ground, terracotta for action, sage for done.
// Calming, boutique-clinic feel rather than a clinical/corporate one.
export const colors = {
  bg: '#fbf3e7',
  card: '#fffdf8',
  text: '#3b2e22',
  muted: '#8a7357',
  accent: '#c9713d',
  accentDark: '#a95c2f',
  accentSoft: '#f3e0c9',
  accentText: '#fffaf3',
  success: '#7c9473',
  successSoft: '#e9efe4',
  line: '#ecd9bf',
  lineSoft: '#f4e9d7',
  danger: '#c0503a',
  dangerSoft: '#f8e3dc',
};

export const fonts = {
  regular: 'Inter_400Regular',
  medium: 'Inter_500Medium',
  semibold: 'Inter_600SemiBold',
  bold: 'Inter_700Bold',
  extrabold: 'Inter_800ExtraBold',
  // Warm humanist serif, used sparingly for titles and branding.
  displaySemibold: 'Lora_600SemiBold',
  displayBold: 'Lora_700Bold',
};

export const radius = {
  sm: 12,
  md: 16,
  lg: 22,
  xl: 30,
  pill: 999,
};

// react-native-web translates these into an equivalent CSS box-shadow, so the
// same tokens work on web and native without a separate implementation.
export const shadow = {
  card: {
    shadowColor: '#4a3624',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 12,
    elevation: 2,
  },
  raised: {
    shadowColor: '#4a3624',
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.16,
    shadowRadius: 26,
    elevation: 8,
  },
};
