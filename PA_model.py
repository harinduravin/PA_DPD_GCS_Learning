import tensorflow as tf
import numpy as np
import sionna as sn
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt

class ModifiedRappModel(tf.keras.layers.Layer):
    """TensorFlow implementation of Modified Rapp Power Amplifier Model"""

    def __init__(self, vsat=0.5, p=1.75, g_r=1.0, q1=1.4, q2=1.4,
                 alpha=-490, beta=0.17):
        super().__init__()

        self.vsat = vsat
        self.p = p
        self.g_r = g_r
        self.q1 = q1
        self.q2 = q2
        self.alpha = alpha
        self.beta = beta

    # @tf.function
    def amplify(self, vin):
        """
        Apply Modified Rapp model amplification
        vin: complex tensor (tf.complex64 or tf.complex128)
        """
        vin = tf.cast(vin, tf.complex64)

        abs_v = tf.abs(vin)

        # -----------------------------
        # AM / AM conversion
        # -----------------------------
        vout_m = (
            self.g_r * abs_v /
            tf.pow(
                1.0 + tf.pow(self.g_r * abs_v / self.vsat, 2.0 * self.p),
                1.0 / (2.0 * self.p)
            )
        )

        # -----------------------------
        # AM / PM conversion
        # -----------------------------
        abs_v_pm = tf.abs(vin / 10.0)

        vout_p = (
            self.alpha * tf.pow(abs_v_pm, self.q1) /
            (1.0 + tf.pow(abs_v_pm, self.q2) / self.beta)
        )

        # -----------------------------
        # Combine AM/AM and AM/PM
        # -----------------------------
        phase = tf.math.angle(vin)
        phase_out = vout_p * (np.pi / 180.0) + phase

        # vout = tf.cast(vout_m, tf.complex64) * tf.exp(
        #     tf.complex(tf.zeros_like(phase_out), phase_out)
        # )

        vout = tf.complex(vout_m, tf.zeros_like(vout_m)) * tf.exp(
            tf.complex(tf.zeros_like(phase_out), phase_out)
        )

        return vout
        # return tf.complex(phase_out, phase_out)

class PAModel(tf.keras.Model):
    def __init__(self, fs, f0, device='cpu'):
        super().__init__()
        self.fs = fs
        self.f0 = f0

        # PA variation parameters
        vsat_var = 0.1
        p_var = 0.1

        # Modified Rapp model parameters
        self.vsat = 0.5 + vsat_var
        self.p = 1.75 + p_var
        self.g_r = 1.0
        self.q1 = 1.4
        self.q2 = 1.4
        self.alpha = -490
        self.beta = 0.17

        # Modified Rapp model (TensorFlow version)
        self.rapp_model = ModifiedRappModel(
            vsat=self.vsat,
            p=self.p,
            g_r=self.g_r,
            q1=self.q1,
            q2=self.q2,
            alpha=self.alpha,
            beta=self.beta
        )

    # @tf.function
    def forward_differentiable(self, x):
        """
        Wiener-like differentiable PA model for training
        Structure: LTI Filter -> Rapp Nonlinearity -> LTI Filter
        """

        n_samples = tf.shape(x)[1]
        n_samples_f = tf.cast(n_samples, tf.float32)

        # Frequency grid
        f_vect = tf.linspace(
            -self.fs / 2.0,
            self.fs / 2.0 - self.fs / n_samples_f,
            n_samples
        ) + self.f0

        # Memory parameters
        alpha_pre  = 0.25
        alpha_post = 0.50

        k1_pre,  k2_pre  = 0.0, 0.50
        k1_post, k2_post = 0.0, 1.00

        # -----------------------
        # Pre-filter
        # -----------------------
        f_norm_pre = (f_vect - self.f0) / (self.fs / 2.0)
        mag_pre = tf.exp(-alpha_pre * tf.square(f_norm_pre))
        phase_pre = k1_pre * f_norm_pre + k2_pre * tf.square(f_norm_pre)
        h_pre = tf.cast(mag_pre, tf.complex64) * tf.exp(
            tf.complex(tf.zeros_like(phase_pre), phase_pre)
        )

        # -----------------------
        # Post-filter
        # -----------------------
        f_norm_post = (f_vect - self.f0) / (self.fs / 2.0)
        mag_post = tf.exp(-alpha_post * tf.square(f_norm_post))
        phase_post = k1_post * f_norm_post + k2_post * tf.square(f_norm_post)
        h_post = tf.cast(mag_post, tf.complex64) * tf.exp(
            tf.complex(tf.zeros_like(phase_post), phase_post)
        )

        # Normalize filters
        h_pre /= tf.complex(tf.sqrt(tf.reduce_mean(tf.abs(h_pre) ** 2)),0.0)
        h_post /= tf.complex(tf.sqrt(tf.reduce_mean(tf.abs(h_post) ** 2)),0.0)

        # -----------------------
        # FFT helpers
        # -----------------------
        def fft(x):
            return tf.signal.fftshift(
                tf.signal.fft(tf.signal.ifftshift(x))
            )

        def ifft(x):
            return tf.signal.fftshift(
                tf.signal.ifft(tf.signal.ifftshift(x))
            )

        # tf.print('h_pre: ', h_pre.numpy())
        # tf.print('h_post: ', h_post.numpy())

        # -----------------------
        # Pre-filtering
        # -----------------------
        x_freq = fft(x) / tf.cast(n_samples, tf.complex64)
        x_freq_pre = h_pre * x_freq
        x_pre = ifft(x_freq_pre) * tf.cast(n_samples, tf.complex64)

        # tf.print('x_freq: ', x_freq.numpy())

        # -----------------------
        # Nonlinearity (Rapp)
        # -----------------------
        x_nonlinear = 100.0 * self.rapp_model.amplify(x_pre / 10.0)


        # -----------------------
        # Post-filtering
        # -----------------------
        x_nl_freq = fft(x_nonlinear) / tf.cast(n_samples, tf.complex64)
        x_freq_post = h_post * x_nl_freq
        yout = ifft(x_freq_post) * tf.cast(n_samples, tf.complex64)

        return yout

class ACPRCalculatorTF:
    """TensorFlow implementation of Adjacent Channel Power Ratio calculation"""

    def __init__(self):
        pass

    def welch_tf(self, x, fs, nperseg=512, noverlap=0):
        """
        Stable Welch PSD using TensorFlow FFT.
        Produces two-sided PSD like SciPy (return_onesided=False).
        """

        # Ensure tensor & complex type
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x)
        x = tf.cast(tf.reshape(x, [-1]), tf.complex64)

        win = tf.signal.hann_window(
            nperseg, periodic=True, dtype=tf.float32
        )
        hop = nperseg - noverlap

        # --- Zero-pad if too short (SciPy behavior) ---
        x_len = tf.shape(x)[0]
        x = tf.cond(
            x_len < nperseg,
            lambda: tf.concat(
                [x, tf.zeros(nperseg - x_len, dtype=x.dtype)], axis=0
            ),
            lambda: x
        )

        # --- Frame the signal like Welch ---
        frames = tf.signal.frame(
            x,
            frame_length=nperseg,
            frame_step=hop
        )  # shape [num_frames, nperseg]

        # Apply window
        frames = frames * tf.cast(win, tf.complex64)

        # --- FFT each frame ---
        X = tf.signal.fft(frames)

        # Power spectral density per frame
        win_power = tf.reduce_sum(win**2)
        Pxx = (tf.abs(X)**2) / (fs * win_power)

        # Average across segments
        Pxx_mean = tf.reduce_mean(Pxx, axis=0)

        # Two-sided frequency vector (matches SciPy)
        freqs = tf.range(
            -nperseg // 2,
            nperseg // 2,
            dtype=tf.float32
        ) * (fs / nperseg)

        Pxx_mean = tf.signal.fftshift(Pxx_mean)


        return freqs, Pxx_mean

    def calculate_acpr_sur(
        self,
        signal,
        fs,
        f0_bb,
        meas_bw_main,
        acpr_offsets,
        meas_bw_acpr
    ):

        f, Pxx = self.welch_tf(
            signal, fs, nperseg=512, noverlap=0
        )
        Pxx_lin = tf.abs(Pxx)

        def band_power(f, Pxx, f_center, BW):
            mask = tf.logical_and(
                f >= f_center - BW / 2,
                f <= f_center + BW / 2
            )
            df = f[1] - f[0]
            return tf.reduce_sum(tf.boolean_mask(Pxx, mask)) * tf.abs(df)

        Pmain = band_power(f, Pxx_lin, f0_bb, meas_bw_main)

        ChP = 10.0 * tf.math.log(Pmain + 1e-30) / tf.math.log(10.0)

        Pacp_lower = band_power(
            f, Pxx_lin, f0_bb - acpr_offsets[0], meas_bw_acpr
        )
        Pacp_upper = band_power(
            f, Pxx_lin, f0_bb + acpr_offsets[0], meas_bw_acpr
        )
        # print(Pacp_upper+Pmain+Pacp_lower)
        acpr1 = 10.0 * tf.math.log(Pacp_lower / Pmain + 1e-30) / tf.math.log(10.0)
        acpr2 = 10.0 * tf.math.log(Pacp_upper / Pmain + 1e-30) / tf.math.log(10.0)

        return acpr1, acpr2, ChP
