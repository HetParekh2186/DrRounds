import { useCallback } from 'react';
import { Pressable, type GestureResponderEvent, type PressableProps } from 'react-native';
import Animated, { useAnimatedStyle, useSharedValue, withSpring } from 'react-native-reanimated';

const AnimatedPressableBase = Animated.createAnimatedComponent(Pressable);

/** A Pressable that shrinks slightly on press for tactile feedback. */
export function ScalePressable({
  style,
  onPressIn,
  onPressOut,
  scaleTo = 0.96,
  ...rest
}: PressableProps & { scaleTo?: number }) {
  const scale = useSharedValue(1);
  const animatedStyle = useAnimatedStyle(() => ({ transform: [{ scale: scale.value }] }));

  const handlePressIn = useCallback(
    (e: GestureResponderEvent) => {
      scale.value = withSpring(scaleTo, { damping: 16, stiffness: 350 });
      onPressIn?.(e);
    },
    [onPressIn, scale, scaleTo]
  );

  const handlePressOut = useCallback(
    (e: GestureResponderEvent) => {
      scale.value = withSpring(1, { damping: 16, stiffness: 350 });
      onPressOut?.(e);
    },
    [onPressOut, scale]
  );

  return (
    <AnimatedPressableBase
      style={[style, animatedStyle]}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      {...rest}
    />
  );
}
