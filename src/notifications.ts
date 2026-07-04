import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowBanner: true,
    shouldShowList: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

/**
 * Ensures exactly one repeating daily reminder is scheduled at the given time
 * (default 8:00 PM). Requests permission on first run. Safe to call on every
 * app launch — it cancels any prior schedule before re-adding.
 */
export async function ensureDailyReminder(hour = 20, minute = 0): Promise<boolean> {
  let { granted } = await Notifications.getPermissionsAsync();
  if (!granted) {
    granted = (await Notifications.requestPermissionsAsync()).granted;
  }
  if (!granted) return false;

  if (Platform.OS === 'android') {
    await Notifications.setNotificationChannelAsync('daily', {
      name: 'Daily check',
      importance: Notifications.AndroidImportance.DEFAULT,
    });
  }

  await Notifications.cancelAllScheduledNotificationsAsync();
  await Notifications.scheduleNotificationAsync({
    content: {
      title: 'End-of-day patient check',
      body: 'Tap to see who you still need to visit today.',
    },
    trigger: {
      type: Notifications.SchedulableTriggerInputTypes.DAILY,
      hour,
      minute,
    },
  });
  return true;
}
