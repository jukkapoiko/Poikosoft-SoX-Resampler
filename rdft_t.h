/* FFT dispatch with exact function-pointer types. */
#ifndef SOXR_RDFT_T_H
#define SOXR_RDFT_T_H
#include <stddef.h>
typedef struct soxr_rdft_callbacks {
  void * (*forward_setup)(int);
  void * (*backward_setup)(int);
  void (*delete_setup)(void *);
  void (*forward)(int, void *, void *, void *);
  void (*oforward)(int, void *, void *, void *);
  void (*backward)(int, void *, void *, void *);
  void (*obackward)(int, void *, void *, void *);
  void (*convolve)(int, void *, void *, void const *);
  void (*convolve_portion)(int, void *, void const *);
  int (*multiplier)(void);
  void (*reorder_back)(int, void *, void *, void *);
  void * (*mem_malloc)(size_t);
  void * (*mem_calloc)(size_t, size_t);
  void (*mem_free)(void *);
  int (*flags)(void);
  int (*oom_status)(void);
  void (*l2_fused)(int, void *, void *, void *, void const *);
} soxr_rdft_callbacks_t;
#define rdft_forward_setup    RDFT_CB->forward_setup
#define rdft_backward_setup   RDFT_CB->backward_setup
#define rdft_delete_setup     RDFT_CB->delete_setup
#define rdft_forward          RDFT_CB->forward
#define rdft_oforward         RDFT_CB->oforward
#define rdft_backward         RDFT_CB->backward
#define rdft_obackward        RDFT_CB->obackward
#define rdft_convolve         RDFT_CB->convolve
#define rdft_convolve_portion RDFT_CB->convolve_portion
#define rdft_multiplier       RDFT_CB->multiplier
#define rdft_reorder_back     RDFT_CB->reorder_back
#define rdft_malloc           RDFT_CB->mem_malloc
#define rdft_calloc           RDFT_CB->mem_calloc
#define rdft_free             RDFT_CB->mem_free
#define rdft_flags            RDFT_CB->flags
#define rdft_oom_status       RDFT_CB->oom_status
#define rdft_l2_fused         RDFT_CB->l2_fused
#define RDFT_IS_SIMD 1
#define RDFT_NEEDS_SCRATCH 2
#define RDFT_L2_FUSED 4
#endif
