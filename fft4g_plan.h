/* Per-stage Ooura FFT plan. The owning resampler frees this at teardown.
 * There is no mutable process-global FFT state or shared FFT cache. */
typedef struct {
  int * ip;
  DFT_FLOAT * w;
  int length;
} soxr_ooura_plan;

static void * ooura_setup(int length)
{
  soxr_ooura_plan * plan;
  if(length < 2 || (length & (length - 1))) return NULL;

  plan = (soxr_ooura_plan *) calloc(1, sizeof(*plan));
  if(!plan) return NULL;

  plan->ip = (int *) calloc(dft_br_len(length), sizeof(*plan->ip));
  plan->w = (DFT_FLOAT *) calloc(dft_sc_len(length), sizeof(*plan->w));
  if(!plan->ip || !plan->w) {
    free(plan->ip);
    free(plan->w);
    free(plan);
    return NULL;
  }

  plan->length = length;
  return plan;
}

static void ooura_delete(void * setup)
{
  soxr_ooura_plan * plan = (soxr_ooura_plan *) setup;
  if(!plan) return;
  free(plan->ip);
  free(plan->w);
  free(plan);
}

static void ooura_transform(int length, void * setup, DFT_FLOAT * samples, int direction)
{
  soxr_ooura_plan * plan = (soxr_ooura_plan *) setup;
  if(!plan || length != plan->length) return;
  OOURA_RDFT(length, direction, samples, plan->ip, plan->w);
}

static int ooura_status(void) { return 0; }
