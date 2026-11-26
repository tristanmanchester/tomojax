// Bidirectional UK/US English spelling normalization for TomoJAX docs search.
// Both the build-time search index (`app/api/search/route.ts`) and client
// queries (`components/search.tsx`) canonicalize tokens to US English so
// British queries match US-spelled docs and US queries still match remaining
// UK-spelled identifiers such as Nikon's `CentreOfRotationTop`.

const WORD_PREFIX =
  "(?:un|re|de|non|pre|post|auto|sub|over|under|mis|co|batch|multi|micro|macro|pseudo|semi)";

// Stems where a bare `<stem>is` typing prefix cannot be a non-UK English word
// (or where non-UK `-ism`/`-ist` words are excluded by the suffix pattern).
const ISE_PREFIX_SAFE_STEMS = [
  "regular",
  "normal",
  "initial",
  "optim",
  "voxel",
  "linear",
  "minim",
  "maxim",
  "discret",
  "serial",
  "visual",
  "recogn",
  "summar",
  "vector",
  "random",
  "parameter",
  "material",
  "external",
  "canonical",
  "contextual",
  "penal",
  "raster",
  "parallel",
  "stabil",
  "standard",
  "custom",
  "local",
  "global",
  "general",
  "special",
  "character",
  "synchron",
  "priorit",
  "quant",
  "apod",
  "homogen",
  "symmetr",
  "polar",
  "ion",
  "binar",
  "equal",
  "digit",
  "ideal",
  "final",
  "util",
  "amort",
  "major",
  "marginal",
  "orthogonal",
  "diagonal",
  "factor",
  "sanit",
  "token",
  "categor",
  "author",
  "memor",
  "central",
  "neutral",
  "modern",
  "automat",
  "systemat",
  "crystall",
  "vapor",
  "pressur",
  "sensit",
  "modular",
  "granular",
  "scalar",
  "tensor",
  "hybrid",
  "harmon",
  "privat",
  "conceptual",
  "actual",
  "virtual",
  "spatial",
  "spectral",
  "temporal",
  "singular",
  "circular",
] as const;

// Stems where bare `<stem>is` is a valid word or prefix of a common non-UK
// word (`realistic`, `emphasis`, `synthesis`, `hypothesis`, `mechanism`,
// `organism`, `criticism`, `formalism`, `theorist`), so require a UK suffix.
const ISE_SUFFIX_ONLY_STEMS = [
  "real",
  "emphas",
  "synthes",
  "hypothes",
  "mechan",
  "organ",
  "critic",
  "formal",
  "theor",
] as const;

const ISE_NONEMPTY_SUFFIX =
  "e(?:d|s|r|rs)?|ing|in|a(?:t(?:i(?:o(?:n(?:s|al)?)?)?)?)?|ab(?:l(?:e|y)|il(?:i(?:t(?:y|ies)?)?)?)?";

const ISE_PREFIX_SAFE_RE = new RegExp(
  `^(${WORD_PREFIX}{0,2}(?:${ISE_PREFIX_SAFE_STEMS.join("|")}))is(|${ISE_NONEMPTY_SUFFIX})$`,
);

const ISE_SUFFIX_ONLY_RE = new RegExp(
  `^(${WORD_PREFIX}{0,2}(?:${ISE_SUFFIX_ONLY_STEMS.join("|")}))is(${ISE_NONEMPTY_SUFFIX})$`,
);

const PARAMETRIS_RE = new RegExp(
  `^(re|over|under|un|non)?parametr(?:is|iz)(|${ISE_NONEMPTY_SUFFIX})$`,
);

const YSE_RE =
  /^(re)?(anal|catal|paral|hydrol|electrol|dial)ys(e(?:d|r|rs)?|ing|in)$/;

const CENTRE_PREFIX =
  "(?:iso|epi|bary|ortho|circum|re|off|sub|mid|auto|automatic)";

const CENTRE_INFLECTED_RE = new RegExp(
  `^(${CENTRE_PREFIX})?centr(?:e(s|d)?|(ing|in))$`,
);

const CENTRE_COMPOUND_RE = new RegExp(`^(${CENTRE_PREFIX})?centre([a-z]+)$`);

const METRE_RE =
  /^((?:sub)?(?:milli|micro|nano|centi|kilo|pico|femto|deci)?)metre(s?)$/;

const OTHER_RE_ENDING_RE = /^((?:milli|micro|centi|deci)?lit|fib|calib)re(s?)$/;

const OUR_STEMS = [
  "col",
  "behavi",
  "neighb",
  "hon",
  "fav",
  "vap",
  "flav",
  "tum",
  "hum",
  "rig",
  "vig",
  "od",
  "lab",
  "harb",
  "arm",
  "splend",
] as const;

const OUR_PREFIX =
  "(?:un|re|dis|mis|en|multi|bi|tri|pseudo|false|true|mono|poly|feature)";

const OUR_STEM_PATTERN = OUR_STEMS.join("|");

const OUR_TYPING_PREFIX_RE = new RegExp(
  `^(${OUR_PREFIX}?)(${OUR_STEM_PATTERN})ou$`,
);

const OUR_FULL_RE = new RegExp(
  `^(${OUR_PREFIX}?)(${OUR_STEM_PATTERN})our(s|ed|ing|in|er|ers|al|ally|able|ably|ite|ites|hood|hoods|less|ful|fully|ly|map|maps|bar|bars|space|spaces)?$`,
);

const DOUBLE_L_RE =
  /^(un|re|mis|pre|sub|over|multi)?(label|model|travel|signal|level|channel|tunnel|panel|fuel)l(|e|ed|er|ers|i|in|ing)$/;

const CANCEL_LL_RE = /^cancell(|e|ed|er|ers|i|in|ing)$/;

const OGUE_RE =
  /^(analog|catalog|dialog|homolog|monolog|prolog|epilog)u(?:e(s|d)?|(ing|in))?$/;

const ARTEFACT_RE =
  /^(simulation|ring|streak|cone|motion|metal|beam)?artef([a-z]*)$/;

const GREY_RE =
  /^(light|dark|mid|blue)?grey(s|scale|ed|ing|ish|level|levels|value|values|tone|tones)?$/;

const LICENCE_RE = /^(un|re)?licenc(e|ed|es|ing|in)?$/;

const ENCE_RE = /^(defen|offen)c(e|es)?$/;

const ANTICLOCKWISE_RE = /^anti-?clock(w(?:i(?:s(?:e)?)?)?)?$/;

const EMENT_RE = /^(acknowledg|judg)ement(s?)$/;

const PRACTISE_RE = /^practis(e|ed|es|ing|in)$/;

const PROGRAMME_RE = /^programme(s?)$/;

const MANOEUVRE_RE = /^manoeuvr(?:e(s|d)?|(ing|in))$/;

// Reverse patterns used only to highlight remaining UK-spelled text (such as
// `CentreOfRotationTop`) when a user searches with US spellings.
const US_CENTER_INFLECTED_RE = new RegExp(
  `^(${CENTRE_PREFIX})?center(s|d|ed|ing|in)?$`,
);
const US_CENTER_COMPOUND_RE = new RegExp(`^(${CENTRE_PREFIX})?center([a-z]+)$`);
const US_IZE_RE = new RegExp(
  `^(${WORD_PREFIX}{0,2}(?:${[...ISE_PREFIX_SAFE_STEMS, ...ISE_SUFFIX_ONLY_STEMS].join("|")}))iz(|${ISE_NONEMPTY_SUFFIX})$`,
);
const US_OR_RE = new RegExp(
  `^(${OUR_PREFIX}?)(${OUR_STEM_PATTERN})or(s|ed|ing|in|er|ers|al|ally|able|ably|ite|ites|hood|hoods|less|ful|fully|ly|map|maps|bar|bars|space|spaces)?$`,
);
const US_ARTIFACT_RE =
  /^(simulation|ring|streak|cone|motion|metal|beam)?artif([a-z]*)$/;
const US_GRAY_RE =
  /^(light|dark|mid|blue)?gray(s|scale|ed|ing|ish|level|levels|value|values|tone|tones)?$/;
const US_METER_RE =
  /^((?:sub)?(?:milli|micro|nano|centi|kilo|pico|femto|deci)?)meter(s?)$/;
const US_SINGLE_L_RE =
  /^(un|re|mis|pre|sub|over|multi)?(label|model|cancel)(ed|er|ers|ing)$/;

function normalizeAlphabeticWord(word: string): string {
  if (word.length < 4) return word;

  let m: RegExpMatchArray | null;

  m = word.match(CENTRE_INFLECTED_RE);
  if (m) {
    const suffix = m[2] === "d" ? "ed" : (m[2] ?? m[3] ?? "");
    return `${m[1] ?? ""}center${suffix}`;
  }

  m = word.match(CENTRE_COMPOUND_RE);
  if (m) return `${m[1] ?? ""}center${m[2]}`;

  m = word.match(METRE_RE);
  if (m) return `${m[1]}meter${m[2]}`;

  m = word.match(OTHER_RE_ENDING_RE);
  if (m) return `${m[1]}er${m[2]}`;

  m = word.match(PARAMETRIS_RE);
  if (m) return `${m[1] ?? ""}parameteriz${m[2]}`;

  m = word.match(ISE_PREFIX_SAFE_RE);
  if (m) return `${m[1]}iz${m[2]}`;

  m = word.match(ISE_SUFFIX_ONLY_RE);
  if (m) return `${m[1]}iz${m[2]}`;

  m = word.match(YSE_RE);
  if (m) return `${m[1] ?? ""}${m[2]}yz${m[3]}`;

  m = word.match(OUR_TYPING_PREFIX_RE);
  if (m) return `${m[1]}${m[2]}or`;

  m = word.match(OUR_FULL_RE);
  if (m) return `${m[1]}${m[2]}or${m[3] ?? ""}`;

  m = word.match(DOUBLE_L_RE);
  if (m) return `${m[1] ?? ""}${m[2]}${m[3]}`;

  m = word.match(CANCEL_LL_RE);
  if (m) return `cancel${m[1]}`;

  m = word.match(OGUE_RE);
  if (m) {
    const suffix = m[2] === "d" ? "ed" : (m[2] ?? m[3] ?? "");
    return `${m[1]}${suffix}`;
  }

  m = word.match(ARTEFACT_RE);
  if (m) return `${m[1] ?? ""}artif${m[2]}`;

  m = word.match(GREY_RE);
  if (m) return `${m[1] ?? ""}gray${m[2] ?? ""}`;

  if (word === "aluminium") return "aluminum";
  if (word === "aluminiu" || word === "alumini") return "aluminu";

  m = word.match(LICENCE_RE);
  if (m) return `${m[1] ?? ""}licens${m[2] ?? ""}`;

  m = word.match(ENCE_RE);
  if (m) return `${m[1]}s${m[2] ?? ""}`;

  m = word.match(ANTICLOCKWISE_RE);
  if (m) return `counterclock${m[1] ?? ""}`;

  m = word.match(EMENT_RE);
  if (m) return `${m[1]}ment${m[2]}`;

  if (word === "zeroes") return "zeros";

  m = word.match(PRACTISE_RE);
  if (m) return `practic${m[1]}`;

  m = word.match(PROGRAMME_RE);
  if (m) return `program${m[1]}`;

  if (word.startsWith("sulphur")) return `sulfur${word.slice(7)}`;
  if (word === "caesium") return "cesium";
  if (word === "ageing") return "aging";

  m = word.match(MANOEUVRE_RE);
  if (m) {
    const suffix = m[1] === "d" ? "ed" : (m[1] ?? m[2] ?? "");
    return `maneuver${suffix}`;
  }

  return word;
}

/**
 * Normalize a single search token (which may include `-`, `_`, or `'`) from UK
 * to canonical US English. Passed to the build-time search tokenizer so both
 * US- and UK-spelled docs share one index representation.
 */
export function normalizeSpellingToken(token: string): string {
  const lower = token.toLowerCase();
  if (!lower) return lower;

  const antiMatch = lower.match(ANTICLOCKWISE_RE);
  if (antiMatch) return `counterclock${antiMatch[1] ?? ""}`;

  if (!/[-_']/.test(lower)) {
    return normalizeAlphabeticWord(lower);
  }

  return lower
    .split(/([-_'])/)
    .map((part, index) =>
      index % 2 === 0 ? normalizeAlphabeticWord(part) : part,
    )
    .join("");
}

/**
 * Normalize a user's search query for lookup without modifying the text shown
 * in the search input box.
 */
export function normalizeSearchQuery(query: string): string {
  return query.replace(/[A-Za-z0-9_'-]+/g, (raw) => {
    const leadMatch = raw.match(/^['-]+/);
    const trailMatch = raw.match(/['-]+$/);
    const lead = leadMatch ? leadMatch[0] : "";
    const trail = trailMatch ? trailMatch[0] : "";
    const coreEnd = raw.length - trail.length;
    if (lead.length >= coreEnd) return raw;
    const core = raw.slice(lead.length, coreEnd);
    return `${lead}${normalizeSpellingToken(core)}${trail}`;
  });
}

function toBritishHighlightWord(word: string): string {
  let m: RegExpMatchArray | null;

  m = word.match(US_CENTER_INFLECTED_RE);
  if (m) {
    const prefix = m[1] ?? "";
    const suffix = m[2] ?? "";
    if (suffix === "ing" || suffix === "in") return `${prefix}centr${suffix}`;
    if (suffix === "ed") return `${prefix}centred`;
    return `${prefix}centre${suffix}`;
  }

  m = word.match(US_CENTER_COMPOUND_RE);
  if (m) return `${m[1] ?? ""}centre${m[2]}`;

  m = word.match(US_IZE_RE);
  if (m) return `${m[1]}is${m[2]}`;

  m = word.match(US_OR_RE);
  if (m) return `${m[1]}${m[2]}our${m[3] ?? ""}`;

  m = word.match(US_ARTIFACT_RE);
  if (m) return `${m[1] ?? ""}artef${m[2]}`;

  m = word.match(US_GRAY_RE);
  if (m) return `${m[1] ?? ""}grey${m[2] ?? ""}`;

  m = word.match(US_METER_RE);
  if (m) return `${m[1]}metre${m[2]}`;

  m = word.match(US_SINGLE_L_RE);
  if (m) return `${m[1] ?? ""}${m[2]}l${m[3]}`;

  if (word === "aluminum") return "aluminium";
  if (word === "license") return "licence";
  if (word === "licenses") return "licences";
  if (word === "analog") return "analogue";
  if (word === "catalog") return "catalogue";
  if (word === "counterclockwise") return "anticlockwise";

  return word;
}

function toBritishHighlightToken(token: string): string {
  const lower = token.toLowerCase();
  if (!/[-_']/.test(lower)) return toBritishHighlightWord(lower);
  return lower
    .split(/([-_'])/)
    .map((part, index) =>
      index % 2 === 0 ? toBritishHighlightWord(part) : part,
    )
    .join("");
}

/**
 * Expand a query with UK/US spelling equivalents solely for result-snippet
 * highlighting (`useHighlightQuery`), leaving the visible input untouched.
 */
export function expandHighlightQuery(query: string): string {
  const terms = query.trim().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return "";

  const expanded: string[] = [];
  const seen = new Set<string>();

  const add = (term: string) => {
    const key = term.toLowerCase();
    if (!key || seen.has(key)) return;
    seen.add(key);
    expanded.push(term);
  };

  for (const term of terms) {
    add(term);
    add(normalizeSearchQuery(term));
    add(toBritishHighlightToken(term));
  }

  return expanded.join(" ");
}
