export interface QuickAddFields {
  name: string;
  hospital: string;
  room: string;
  note: string;
}

const NO_NOTE_RE = /\b(?:and\s+)?no\s+(?:comments?|notes?)\b\.?/i;
const ROOM_RE = /\b(?:room|bed)\b\s*(?:number\s*)?(?:is\s*|#\s*)?(\d+\w*)/i;
const NOTE_RE = /\b(?:note|comments?)\b\s*(?:is|are)?\s*:?\s*(.+)$/i;
// The negative lookahead right after "is"/"at"/"in" rejects a capture that IS
// itself a stop word — otherwise "hospital room and" (no name actually said)
// would grab "room" as a fake hospital name, since nothing else stopped it.
const HOSPITAL_RE =
  /\bhospital(?:'s)?\b\s*(?:is\s*)?(?!(?:and|room|bed|note|comments?|hospitals?)\b)([a-z][a-z .'-]*?)(?=\s+\b(?:and|room|bed|note|comments?|hospitals?)\b|,|\.|$)/i;
// Covers the reversed phrasing "at/in <name> hospital(s)" (hospital word said last).
const HOSPITAL_BEFORE_RE =
  /\b(?:at|in)\s+(?!(?:the|a|an|and|room|bed|note|comments?)\b)([a-z][a-z .'-]*?)\s+hospitals?\b/i;
const NAME_PREFIX_RE =
  /^(?:add\s+a?\s*(?:new\s+)?patient\s+(?:by\s+the\s+name\s+(?:of\s+)?|by\s+name\s+|named|whose\s+name\s+is|name\s+is|name)|add\s+a?\s*(?:new\s+)?patient|new\s+patient|patient(?:'s)?\s*name\s*is|patient\s*is|by\s+the\s+name\s+(?:of\s+)?|by\s+name\s+|named|name\s*(?:is|of)?)\s*/i;
const FILLER_WORDS_RE = /\b(?:and|his|her|its|the)\b/gi;

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function clean(s: string): string {
  return s
    .replace(FILLER_WORDS_RE, ' ')
    .replace(/^[\s,.]+|[\s,.]+$/g, '')
    .replace(/[,]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Parses one dictated sentence into name/hospital/room/note. Doctors dictate
 * naturally ("his hospital is Apollo, room number is 204, no comments"),
 * without saying "comma" between parts — so this looks for spoken keyword
 * anchors (hospital/room/bed/note/comment) by position, rather than
 * requiring punctuation. Text before the first recognized anchor is the
 * name; text after the last one (if nothing else claimed it) is the note.
 * Falls back to comma-splitting only when no keyword is found at all, for
 * anyone who prefers the terser shorthand.
 */
export function parseQuickAddText(raw: string, knownHospitals: string[]): QuickAddFields {
  const text = raw.trim();
  const hasAnyKeyword = /\b(?:hospital|room|bed|note|comments?)\b/i.test(text);

  if (!hasAnyKeyword && text.includes(',')) {
    return parseCommaShorthand(text, knownHospitals);
  }

  const spans: Array<{ start: number; end: number }> = [];
  let noteExplicitlyEmpty = false;
  let room = '';
  let note = '';
  let hospital = '';

  const noNoteMatch = text.match(NO_NOTE_RE);
  if (noNoteMatch) {
    noteExplicitlyEmpty = true;
    spans.push({ start: noNoteMatch.index!, end: noNoteMatch.index! + noNoteMatch[0].length });
  }

  const roomMatch = text.match(ROOM_RE);
  if (roomMatch) {
    room = roomMatch[1];
    spans.push({ start: roomMatch.index!, end: roomMatch.index! + roomMatch[0].length });
  }

  if (!noteExplicitlyEmpty) {
    const noteMatch = text.match(NOTE_RE);
    if (noteMatch && noteMatch[1].trim()) {
      note = clean(noteMatch[1]);
      spans.push({ start: noteMatch.index!, end: text.length });
    }
  }

  const hospitalMatch = text.match(HOSPITAL_RE) ?? text.match(HOSPITAL_BEFORE_RE);
  if (hospitalMatch && hospitalMatch[1].trim()) {
    const captured = hospitalMatch[1].trim();
    const known = knownHospitals.find((h) => h.toLowerCase() === captured.toLowerCase());
    hospital = known ?? captured;
    spans.push({ start: hospitalMatch.index!, end: hospitalMatch.index! + hospitalMatch[0].length });
  }

  let name: string;
  if (spans.length === 0) {
    name = clean(text.replace(NAME_PREFIX_RE, ''));
  } else {
    spans.sort((a, b) => a.start - b.start);
    const firstStart = spans[0].start;
    const lastEnd = spans[spans.length - 1].end;
    name = clean(text.slice(0, firstStart).replace(NAME_PREFIX_RE, ''));

    if (!note && !noteExplicitlyEmpty) {
      const trailing = clean(text.slice(lastEnd));
      if (trailing) note = trailing;
    }
  }

  // A comma-shorthand hospital (no literal word "hospital" said) may still
  // have landed in name/note — recover it if it matches a known hospital.
  if (!hospital) {
    for (const h of knownHospitals) {
      const re = new RegExp(`\\b${escapeRegExp(h)}\\b`, 'i');
      if (re.test(name)) {
        hospital = h;
        name = clean(name.replace(re, ''));
        break;
      }
      if (re.test(note)) {
        hospital = h;
        note = clean(note.replace(re, ''));
        break;
      }
    }
  }

  return { name, hospital, room, note };
}

function parseCommaShorthand(raw: string, knownHospitals: string[]): QuickAddFields {
  const segments = raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);

  const name = segments.shift() ?? '';
  const remaining: string[] = [];
  let room = '';

  for (const segment of segments) {
    const match = segment.match(/\b(?:room|bed)\b\s*#?\s*(\S+)|^#?\s*(\d+\w*)$/i);
    if (!room && match) {
      room = match[1] ?? match[2] ?? '';
    } else {
      remaining.push(segment);
    }
  }

  let hospital = '';
  const leftover: string[] = [];
  for (const segment of remaining) {
    const known = !hospital
      ? knownHospitals.find((h) => h.toLowerCase() === segment.toLowerCase())
      : undefined;
    if (known) {
      hospital = known;
    } else {
      leftover.push(segment);
    }
  }

  return { name, hospital, room, note: leftover.join(', ') };
}
