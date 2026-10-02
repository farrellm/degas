// What the fake server answers with: an SDXL family by default, Wan 2.2 where a test adds it.

export const FAMILIES = [
  {
    id: 'sdxl',
    label: 'Stable Diffusion XL',
    media: 'image',
    lora_format: 'single',
    variants: [
      {
        id: 'base',
        label: 'SDXL',
        min_gpu: 'T4',
        modes: ['t2i'],
        size_constraints: {
          multiple_of: 8,
          min_pixels: 262144,
          max_pixels: 2359296,
          presets: [
            [1024, 1024],
            [832, 1216],
          ],
        },
      },
    ],
  },
]

export const SCHEMA = {
  type: 'object',
  required: ['prompt'],
  properties: {
    prompt: { type: 'string', title: 'Prompt', 'x-widget': 'prompt' },
    width: { type: 'integer', title: 'Width', default: 1024, 'x-widget': 'aspect' },
    height: { type: 'integer', title: 'Height', default: 1024, 'x-widget': 'aspect' },
    steps: {
      type: 'integer',
      title: 'Steps',
      default: 30,
      minimum: 1,
      maximum: 100,
      'x-widget': 'slider',
    },
    seed: { type: 'integer', title: 'Seed', default: -1, 'x-widget': 'seed' },
    scheduler: {
      type: 'string',
      title: 'Sampler',
      default: 'euler',
      enum: ['euler', 'ddim'],
      'x-enum-labels': ['Euler', 'DDIM'],
      'x-widget': 'select',
      'x-advanced': true,
    },
    vae_fp32: {
      type: 'boolean',
      title: 'Built-in VAE in float32',
      description: "The checkpoint's own VAE instead of the fp16 fix. Slower.",
      default: false,
      'x-advanced': true,
    },
  },
}

export function SESSION_BASE() {
  return {
    session: null,
    step: null,
    worker: null,
    idle_deadline: null,
    idle_timeout_min: 15,
    drive: { configured: true, authorized: true, error: null, push_error: null },
    gpus: ['T4', 'L4'],
  }
}

export const SESSION = SESSION_BASE()

export const ASSETS = [
  {
    path: 'models/sdxl/juggernaut.safetensors',
    family: 'sdxl',
    kind: 'model',
    size: 6_938_040_682,
    sidecar: { label: 'Juggernaut XL v10' },
    preview_thumb: null,
  },
  {
    path: 'models/sdxl/base.safetensors',
    family: 'sdxl',
    kind: 'model',
    size: 6_938_040_682,
    sidecar: null,
    preview_thumb: null,
  },
  {
    path: 'loras/sdxl/film.safetensors',
    family: 'sdxl',
    kind: 'lora',
    size: 144_000_000,
    sidecar: { label: 'Film Grain v3', trigger_words: ['filmgrain'], default_weight: 0.8 },
    preview_thumb: 'abc',
  },
]

export const RUNNING = {
  ...SESSION_BASE(),
  session: {
    id: 's1',
    gpu: 'L4',
    high_mem: false,
    state: 'ready',
    started_at: '2026-09-27T12:00:00Z',
    ended_at: null,
    last_activity_at: '2026-09-27T12:00:00Z',
    error: null,
  },
  worker: {
    gpu: 'NVIDIA L4',
    vram_free: 20e9,
    vram_total: 24e9,
    disk_free: 150e9,
    cache: {
      used: 6_938_040_682,
      budget: 150e9,
      files: [{ path: 'models/sdxl/juggernaut.safetensors', size: 6_938_040_682, last_used: 1 }],
    },
  },
}

export const SPEC = {
  family: 'sdxl',
  variant: 'base',
  mode: 't2i',
  model: { path: 'models/sdxl/juggernaut.safetensors' },
  params: {
    prompt: 'a lighthouse',
    width: 832,
    height: 1216,
    steps: 30,
    seed: -1,
    scheduler: 'euler',
  },
}

export const JOB_DONE = {
  id: 'j1',
  status: 'done',
  spec: SPEC,
  seeds: [1234],
  created_at: '2026-09-27T12:00:00Z',
  progress: null,
  error: null,
}

export const RESULT = {
  id: 'r1',
  job_id: 'j1',
  item_index: 0,
  blob_sha: 'abc',
  media_type: 'image/png',
  seed: 1234,
  width: 832,
  height: 1216,
  created_at: '2026-09-27T12:00:00Z',
  expires_at: null,
  library_id: null,
  spec: SPEC,
}

export const LIBRARY_ITEM = {
  id: 'k1',
  kind: 'image',
  blob_sha: 'abc',
  media_type: 'image/png',
  width: 832,
  height: 1216,
  config: {
    ...SPEC,
    degas_version: 1,
    model: { path: 'models/sdxl/retired.safetensors' },
    loras: [{ path: 'loras/sdxl/film.safetensors', weight: 0.6 }],
    params: { ...SPEC.params, prompt: 'a harbour at dusk', seed: 77 },
  },
  title: null,
  tags: ['sea'],
  created_at: new Date().toISOString(),
  source_result_id: null,
}

/** Two image prompts: a pair of style pictures, and a face. */
export const IMAGE_PROMPTS = [
  {
    adapter: { path: 'ip_adapters/sdxl/ip-adapter-plus_sdxl_vit-h.safetensors' },
    images: ['sha256:pic1', 'sha256:pic2'],
    fit: 'crop',
    purpose: 'style',
    weight: 1,
    start: 0,
    end: 0.8,
  },
  {
    adapter: { path: 'ip_adapters/sdxl/ip-adapter-faceid-plusv2_sdxl.bin' },
    images: ['sha256:face1'],
    fit: 'crop',
    purpose: 'all',
    weight: 0.8,
    start: 0,
    end: 1,
    structure: 1,
    lora_weight: 0.6,
  },
]

export const PROMPT = {
  id: 'p1',
  name: 'Harbour',
  prompt: 'a harbour at dusk, oil painting',
  negative_prompt: '',
  family: 'sdxl',
  tags: [],
  created_at: '2026-09-27T12:00:00Z',
}

export const WAN = {
  id: 'wan22',
  label: 'Wan 2.2',
  media: 'video',
  lora_format: 'paired_hi_lo',
  variants: [
    {
      id: 'ti2v-5b',
      label: 'Wan 2.2 TI2V 5B',
      min_gpu: 'L4',
      modes: ['t2v', 'i2v'],
      model_dir: 'models/wan22/ti2v-5b',
      lora_format: 'single',
      size_constraints: {
        multiple_of: 32,
        min_pixels: 399360,
        max_pixels: 942080,
        presets: [
          [1280, 704],
          [704, 1280],
        ],
      },
    },
    {
      id: 't2v-a14b',
      label: 'Wan 2.2 T2V A14B',
      min_gpu: 'A100',
      modes: ['t2v'],
      model_dir: 'models/wan22/t2v-a14b',
      lora_format: 'paired_hi_lo',
      size_constraints: {
        multiple_of: 16,
        min_pixels: 230400,
        max_pixels: 921600,
        presets: [
          [1280, 720],
          [832, 480],
        ],
      },
    },
  ],
}

export const WAN_SCHEMA = {
  type: 'object',
  required: ['prompt'],
  properties: {
    prompt: { type: 'string', title: 'Prompt', 'x-widget': 'prompt' },
    width: { type: 'integer', title: 'Width', default: 1280, 'x-widget': 'aspect' },
    height: { type: 'integer', title: 'Height', default: 704, 'x-widget': 'aspect' },
    num_frames: {
      type: 'integer',
      title: 'Frames',
      default: 121,
      minimum: 17,
      maximum: 121,
      'x-step': 4,
      'x-widget': 'slider',
    },
    fps: { type: 'integer', title: 'Frame rate', default: 24, 'x-widget': 'slider' },
    steps: { type: 'integer', title: 'Steps', default: 50, 'x-widget': 'slider' },
    cfg: { type: 'number', title: 'CFG', default: 5, 'x-widget': 'slider' },
    seed: { type: 'integer', title: 'Seed', default: -1, 'x-widget': 'seed' },
  },
}

export const WAN_ASSETS = [
  {
    path: 'models/wan22/ti2v-5b',
    family: 'wan22',
    kind: 'model',
    size: 32e9,
    sidecar: { label: 'TI2V 5B' },
    preview_thumb: null,
  },
  {
    path: 'models/wan22/t2v-a14b/Wan2.2-T2V-A14B',
    family: 'wan22',
    kind: 'model',
    size: 120e9,
    sidecar: null,
    preview_thumb: null,
  },
  ...['motion_high_noise', 'motion_low_noise', 'grain'].map((name) => ({
    path: `loras/wan22/${name}.safetensors`,
    family: 'wan22',
    kind: 'lora',
    size: 300e6,
    sidecar: null,
    preview_thumb: null,
  })),
]

export const WAN_SPEC = {
  family: 'wan22',
  variant: 'ti2v-5b',
  mode: 't2v',
  model: { path: 'models/wan22/ti2v-5b' },
  params: {
    prompt: 'waves on the harbour wall',
    width: 1280,
    height: 704,
    num_frames: 121,
    fps: 24,
    steps: 50,
    cfg: 5,
    seed: 5,
  },
}

export const VIDEO_RESULT = {
  ...RESULT,
  id: 'v1',
  job_id: 'j9',
  blob_sha: 'vid',
  media_type: 'video/mp4',
  seed: 5,
  width: 1280,
  height: 704,
  duration: 5.04,
  segments: null,
  spec: WAN_SPEC,
}
