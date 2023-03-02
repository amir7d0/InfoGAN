import tensorflow as tf
from infogan.utils import sample
from infogan.distributions import GaussianNLLLoss, LogProb


class InfoGAN(tf.keras.models.Model):
    def __init__(self, generator, discriminator, recognition,
                 latent_spec, discrete_reg_coeff=1.0, continuous_reg_coeff=1.0):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.recognition = recognition

        self.latent_spec = latent_spec
        self.noise_var = latent_spec['noise-variables']
        self.cont_latent_dist = latent_spec['continuous-latent-codes']  # list of continuous latent codes
        self.disc_latent_dist = latent_spec['discrete-latent-codes']  # list of discrete latent codes

        self.lambda_disc = discrete_reg_coeff
        self.lambda_cont = continuous_reg_coeff
        self.continuous_loss = GaussianNLLLoss()
        self.log_prob = LogProb()
        self.g_optimizer, self.d_optimizer = None, None
        self.loss_fn = None

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(InfoGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    @tf.function()
    def train_step(self, data):
        real_images = data
        batch_size = tf.shape(real_images)[0]
        noise_inputs, disc_inputs, cont_inputs = sample(self.latent_spec, batch_size)

        with tf.GradientTape(persistent=True) as gtape, tf.GradientTape(persistent=True) as dtape:
            fake_images = self.generator(noise_inputs)
            real_d, _ = self.discriminator(real_images)
            fake_d, mid = self.discriminator(fake_images)
            disc_outputs, cont_outputs = self.recognition(mid)

            dis_loss = d_loss = self.loss_fn(tf.ones_like(real_d), real_d) + self.loss_fn(tf.zeros_like(fake_d), fake_d)
            gen_loss = g_loss = self.loss_fn(tf.ones_like(fake_d), fake_d)
            mi_est, cross_ent = 0, 0
            for idx, (cont_input, cont_output) in enumerate(zip(cont_inputs, cont_outputs)):
                # z, z_mean, z_log_var = cont_output
                # continuous_loss = self.continuous_loss(cont_input, z_mean, z_log_var)
                # gen_loss += self.lambda_cont * continuous_loss
                """"########## START ##########"""
                cont_reg_z = cont_input
                cont_reg_dist_info = cont_output[1:]
                """ calculate log probability for input_dist(Uniform) and
                    a normal distribution predicted by Q network (mean=z_mean, var=e^z_log_var)
                """
                cont_log_q_c_given_x = self.log_prob(cont_reg_z, cont_reg_dist_info[0], cont_reg_dist_info[1])
                """ Calculate log probability for input_dist(Uniform) and
                    a normal distribution (mean=0, var=1)"""
                mean, log_var = tf.zeros_like(cont_reg_dist_info[0]), tf.ones_like(cont_reg_dist_info[1])
                cont_log_q_c = self.log_prob(cont_reg_z, mean, log_var)
                cont_cross_ent = tf.reduce_mean(-cont_log_q_c_given_x)
                cont_ent = tf.reduce_mean(-cont_log_q_c)
                cont_mi_est = cont_ent - cont_cross_ent
                mi_est += cont_mi_est
                cross_ent += cont_cross_ent
                gen_loss -= self.lambda_cont * cont_mi_est
                dis_loss -= self.lambda_cont * cont_mi_est
                """########## END ##########"""

            for idx, (disc_input, disc_output) in enumerate(zip(disc_inputs, disc_outputs)):
                # discrete_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(disc_input, disc_output)
                # gen_loss += self.lambda_disc * discrete_loss
                """########## START ##########"""
                input_dist = self.disc_latent_dist[idx]
                disc_reg_z = disc_input
                disc_reg_dist_info = tf.keras.activations.softmax(disc_output)
                """ Calculate log probability for input_dist(Categorical) and
                    a categorical distribution predicted by Q network (disc_reg_z)"""
                disc_log_q_c_given_x = tf.reduce_sum(tf.math.log(disc_reg_dist_info + 1e-8) * disc_reg_z, axis=1)
                prob = tf.ones([batch_size, input_dist.dim]) * (1.0 / input_dist.dim)
                """ Calculate log probability for input_dist(Categorical) and
                    a categorical distribution (c ∼ Cat(K, p))"""
                disc_log_q_c = tf.reduce_sum(tf.math.log(prob + 1e-8) * disc_reg_z, axis=1)
                disc_cross_ent = tf.reduce_mean(-disc_log_q_c_given_x)
                disc_ent = tf.reduce_mean(-disc_log_q_c)
                disc_mi_est = disc_ent - disc_cross_ent
                mi_est += disc_mi_est
                cross_ent += disc_cross_ent
                gen_loss -= self.lambda_disc * disc_mi_est
                dis_loss -= self.lambda_disc * disc_mi_est
                """########## END ##########"""

        g_vars = self.generator.trainable_weights + self.recognition.trainable_weights
        g_grads = gtape.gradient(gen_loss, g_vars)
        self.g_optimizer.apply_gradients(zip(g_grads, g_vars))

        d_vars = self.discriminator.trainable_weights
        d_grads = dtape.gradient(dis_loss, d_vars)
        self.d_optimizer.apply_gradients(zip(d_grads, d_vars))

        return {"G_loss": g_loss, "D_loss": d_loss,
                "MI": mi_est, "Cross_Ent": cross_ent}
