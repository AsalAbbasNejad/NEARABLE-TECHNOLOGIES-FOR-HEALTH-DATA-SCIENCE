export interface Summary {
  motion_class: string | null
  motion_confidence: number | null
  consensus_hr: number | null
  hr_green_fft: number | null
  hr_pos_fft: number | null
  hr_green_peak: number | null
  hr_pos_peak: number | null
  peak_guidance_hr: number | null
  rr_final: number | null
  rr_welch_bpm: number | null
  rr_hilbert_bpm: number | null
  green_reliability: number | null
  pos_reliability: number | null
  green_window_mad: number | null
  pos_window_mad: number | null
  sampling_rate: number | null
  duration: number | null
  selected_pipeline: string | null
  motion_peak_corrected: boolean | null
  left_right_corrected: boolean | null
  left_right_low_lock_corrected: boolean | null
  left_right_high_hr_corrected: boolean | null
  plots: string[]
  log?: string
}

export interface AnalysisResult {
  run_id: string
  summary: Summary
  plot_urls: string[]
  log_url: string | null
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export function analyzeUpload(
  video: File,
  csv: File | null,
): Promise<AnalysisResult> {
  const form = new FormData()
  form.append('video', video)
  if (csv) form.append('csv', csv)
  return fetch('/api/analyze', { method: 'POST', body: form }).then((r) =>
    asJson<AnalysisResult>(r),
  )
}

export interface SerialPort {
  device: string
  description: string
  hwid: string
  candidate: boolean
}

export interface Devices {
  serial_ports: SerialPort[]
  cms_candidate: string | null
  default_camera_index: number
}

export function listDevices(): Promise<Devices> {
  return fetch('/api/devices').then((r) => asJson<Devices>(r))
}

export interface MotionSegment {
  id: number
  motion_class: string
  start_sec: number
  end_sec: number
  duration_sec: number
  hr_median: number | null
  hr_mean: number | null
  hr_count: number
  device_hr_median: number | null
  device_hr_count: number
  rr_median: number | null
  rr_count: number
  green_reliability_median: number | null
  pos_reliability_median: number | null
  low_samples: boolean
}

export interface LiveEvaluation {
  available: boolean
  has_ground_truth: boolean
  reason?: string
  run_id?: string
  summary?: Record<string, number | boolean | null>
  segments?: Array<Record<string, number | string | null>>
  dashboard?: string
  segment_summary_csv?: string
  metric_summary_csv?: string
  log?: string
  urls?: Record<string, string>
}

// Live per-frame state pushed over the WebSocket.
export interface LiveState {
  type: 'started' | 'frame' | 'done' | 'error'
  detail?: string
  cms_connected?: boolean
  cms_streaming?: boolean
  cms_error?: string | null
  serial_port?: string | null
  recording?: boolean
  segment_mode?: boolean
  segments?: MotionSegment[]
  evaluation?: LiveEvaluation
  evaluation_error?: string
  // frame fields
  frame?: string
  frame_index?: number
  elapsed?: number
  fps?: number | null
  face_detected?: boolean
  buffer_seconds?: number
  warming_up?: boolean
  motion_class?: string
  consensus_hr?: number | null
  hr_green_fft?: number | null
  hr_pos_fft?: number | null
  hr_green_peak?: number | null
  hr_pos_peak?: number | null
  green_reliability?: number | null
  pos_reliability?: number | null
  rr_final?: number | null
  sampling_rate?: number | null
  spo2?: number | null
  hardware_hr?: number | null
  // motion-segment mode extras
  segment_motion_class?: string | null
  position_state?: string | null
  position_segment_id?: number | null
  motion_segment_id?: number | null
  position_shift?: number | null
  recent_motion_level?: number | null
  n_motion_segments?: number | null
}
