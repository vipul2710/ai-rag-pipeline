// A Pikachu-style electric mouse running in place, drawn with SMIL so it
// animates inside the desktop's sandboxed SVG frame (no script needed).

const STEP = '0.32s'
const OUTLINE = '#3b2b10'
const YELLOW = '#FFD93B'

const leg = (cx: number, delay: string, fill: string) => `
  <ellipse cx="${cx}" cy="61" rx="4" ry="6.5" fill="${fill}" stroke="${OUTLINE}" stroke-width="1.2">
    <animateTransform attributeName="transform" type="rotate" values="-35 ${cx} 55;35 ${cx} 55;-35 ${cx} 55"
      dur="${STEP}" begin="${delay}" repeatCount="indefinite"/>
  </ellipse>`

const speedLine = (y: number, x: number, len: number, delay: string) => `
  <line x1="${x}" y1="${y}" x2="${x + len}" y2="${y}" stroke="#9aa3ad" stroke-width="2" stroke-linecap="round" opacity="0">
    <animate attributeName="opacity" values="0;0.9;0" dur="0.45s" begin="${delay}" repeatCount="indefinite"/>
    <animateTransform attributeName="transform" type="translate" values="8 0;-8 0" dur="0.45s" begin="${delay}" repeatCount="indefinite"/>
  </line>`

const RAW_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -6 120 78" width="120" height="78">
  <defs>
    <linearGradient id="earA" gradientUnits="userSpaceOnUse" x1="79" y1="22" x2="64" y2="2">
      <stop offset="0" stop-color="${YELLOW}"/><stop offset="0.6" stop-color="${YELLOW}"/>
      <stop offset="0.6" stop-color="#1d1d1d"/><stop offset="1" stop-color="#1d1d1d"/>
    </linearGradient>
    <linearGradient id="earB" gradientUnits="userSpaceOnUse" x1="87" y1="22" x2="96" y2="0">
      <stop offset="0" stop-color="${YELLOW}"/><stop offset="0.6" stop-color="${YELLOW}"/>
      <stop offset="0.6" stop-color="#1d1d1d"/><stop offset="1" stop-color="#1d1d1d"/>
    </linearGradient>
  </defs>

  ${speedLine(38, 4, 14, '0s')}
  ${speedLine(48, 0, 18, '0.15s')}
  ${speedLine(58, 6, 12, '0.3s')}

  <g>
    <animateTransform attributeName="transform" type="translate" values="0 0;0 -3;0 0"
      dur="${STEP}" repeatCount="indefinite"/>

    <!-- lightning tail -->
    <path d="M44 46 L32 38 L38 34 L24 27 L31 23 L14 9 L37 22 L31 27 L44 36 Z"
      fill="${YELLOW}" stroke="${OUTLINE}" stroke-width="1.2" stroke-linejoin="round">
      <animateTransform attributeName="transform" type="rotate" values="-8 44 42;8 44 42;-8 44 42"
        dur="${STEP}" repeatCount="indefinite"/>
    </path>
    <path d="M44 46 L38 42 L41 38 L44 40 Z" fill="#8a5a1a"/>

    ${leg(50, '0.16s', '#e9c22f')}
    ${leg(70, '0.16s', '#e9c22f')}

    <!-- body -->
    <ellipse cx="62" cy="50" rx="20" ry="13" fill="${YELLOW}" stroke="${OUTLINE}" stroke-width="1.2"/>
    <line x1="55" y1="38" x2="57" y2="43" stroke="#8a5a1a" stroke-width="3" stroke-linecap="round"/>
    <line x1="62" y1="37.5" x2="63.5" y2="42" stroke="#8a5a1a" stroke-width="3" stroke-linecap="round"/>

    ${leg(46, '0s', YELLOW)}
    ${leg(74, '0s', YELLOW)}

    <!-- ears, then head -->
    <path d="M84 21 Q88 8 96 0 Q94 12 90 23 Z" fill="url(#earB)" stroke="${OUTLINE}" stroke-width="1.2"/>
    <path d="M75 23 Q66 12 64 2 Q74 9 83 21 Z" fill="url(#earA)" stroke="${OUTLINE}" stroke-width="1.2"/>
    <circle cx="84" cy="34" r="14" fill="${YELLOW}" stroke="${OUTLINE}" stroke-width="1.2"/>

    <!-- face -->
    <circle cx="90" cy="30.5" r="2.7" fill="#1d1d1d"/>
    <circle cx="90.9" cy="29.6" r="0.9" fill="#fff"/>
    <circle cx="96.6" cy="33" r="0.9" fill="#1d1d1d"/>
    <path d="M92.5 36.5 q1.6 1.6 3.2 0" fill="none" stroke="#1d1d1d" stroke-width="1" stroke-linecap="round"/>
    <circle cx="91" cy="40" r="3.6" fill="#E8423B"/>

    <!-- front arm -->
    <ellipse cx="82" cy="51" rx="3" ry="5" fill="${YELLOW}" stroke="${OUTLINE}" stroke-width="1.2">
      <animateTransform attributeName="transform" type="rotate" values="30 82 47;-30 82 47;30 82 47"
        dur="${STEP}" repeatCount="indefinite"/>
    </ellipse>

    <!-- cheek sparks -->
    <path d="M100 38 L106 35 L103 40 L110 38" fill="none" stroke="#FFC400" stroke-width="1.8"
      stroke-linejoin="round" stroke-linecap="round" opacity="0">
      <animate attributeName="opacity" values="0;1;0;0;0" dur="0.9s" repeatCount="indefinite"/>
    </path>
    <path d="M99 44 L105 46 L100 48 L107 51" fill="none" stroke="#FFC400" stroke-width="1.8"
      stroke-linejoin="round" stroke-linecap="round" opacity="0">
      <animate attributeName="opacity" values="0;0;0;1;0" dur="0.9s" repeatCount="indefinite"/>
    </path>
  </g>

  <!-- ground dust -->
  <circle cx="40" cy="68" r="2" fill="#c9b98f" opacity="0">
    <animate attributeName="opacity" values="0.8;0" dur="${STEP}" repeatCount="indefinite"/>
    <animate attributeName="cx" values="44;30" dur="${STEP}" repeatCount="indefinite"/>
  </circle>
</svg>`

// Comments and indentation dropped: less to send and parse on every redraw.
export const PIKACHU_SVG = RAW_SVG
  .replace(/<!--[\s\S]*?-->/g, '')
  .replace(/>\s+</g, '><')
  .replace(/\s+/g, ' ')
  .trim()
