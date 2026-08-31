import 'react-native-url-polyfill/auto';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { createClient } from '@supabase/supabase-js';
import { Platform } from 'react-native';

const url = process.env.EXPO_PUBLIC_SUPABASE_URL!;
const anonKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(url, anonKey, {
  auth: {
    // AsyncStorage works on native; on web, supabase-js defaults to
    // localStorage automatically when no storage adapter is given.
    storage: Platform.OS === 'web' ? undefined : AsyncStorage,
    autoRefreshToken: true,
    persistSession: true,
    // Needed on web so a password-recovery link's token in the URL gets
    // turned into a session automatically when it opens /reset-password.
    detectSessionInUrl: Platform.OS === 'web',
  },
});
