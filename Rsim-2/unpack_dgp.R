library(tidyverse)

d1p <- read_csv('~/projects/irt-position/Rsim/data_person_params.csv')
d1i <- read_csv('~/projects/irt-position/Rsim/data_item_params.csv')

d2p <- read_csv('~/projects/irt-position/Rsim/data2_person_params.csv')
d2i <- read_csv('~/projects/irt-position/Rsim/data2_item_params.csv')

d3p <- read_csv('~/projects/irt-position/Rsim/data3_person_params.csv')
d3i <- read_csv('~/projects/irt-position/Rsim/data3_item_params.csv')

# person

#

d1p %>%
  ggplot(aes(x= theta, y=k)) +
  geom_point()

d2p %>%
  ggplot(aes(x= theta, y=k)) +
  geom_point()

d3p %>%
  ggplot(aes(x= theta, y=k)) +
  geom_point()

# c and k dont seem to be correlated

d1p %>%
  ggplot(aes(x=c, y=k)) +
  geom_point()

d2p %>%
  ggplot(aes(x=c, y=k)) +
  geom_point()

d3p %>%
  ggplot(aes(x=c, y=k)) +
  geom_point()

# c doesn't seem to be correlated with theta

d1p %>%
  ggplot(aes(x=theta, y=c)) +
  geom_point()

d2p %>%
  ggplot(aes(x=theta, y=c)) +
  geom_point()

d3p %>%
  ggplot(aes(x=theta, y=c)) +
  geom_point()


# items


d1i %>%
  ggplot(aes(x=beta_e, y=beta_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()

d2i %>%
  ggplot(aes(x= beta_e, y=beta_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()

d3i %>%
  ggplot(aes(x= beta_e, y=beta_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()

m <- lm(beta_l ~ beta_e, data=d3i)
summary(m)

d1i %>%
  ggplot(aes(x=alpha_e, y=alpha_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()

d2i %>%
  ggplot(aes(x= alpha_e, y=alpha_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()

d3i %>%
  ggplot(aes(x= alpha_e, y=alpha_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()


# ok, what about fitting stuff??

d2pe <- read_csv('~/projects/irt-position/Rsim/data2_experimental_person_params.csv')
d2ie <- read_csv('~/projects/irt-position/Rsim/data2_experimental_item_params.csv')

d2p_full <- full_join(d2p, d2pe, by=c('id', 'sid'), suffix=c('_s', '_l'))

d2p_full %>%
  ggplot(aes(x=theta_s, y=theta_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()

d2p_full %>%
  ggplot(aes(x=k_s, y=k_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()

d2p_full %>%
  ggplot(aes(x=c_s, y=c_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()

d2i_full <- full_join(d2i, d2ie, by=c('itemkey', 'ik'), suffix=c('_s', '_l'))

d2i_full %>%
  ggplot(aes(x=beta_e_s, y=beta_e_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()

d2i_full %>%
  ggplot(aes(x=beta_l_s, y=beta_l_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()

d2i_full %>%
  ggplot(aes(x=alpha_e_s, y=alpha_e_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()

d2i_full %>%
  ggplot(aes(x=alpha_l_s, y=alpha_l_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()


d2ie %>%
  ggplot(aes(x= beta_e, y=beta_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()

d2ie %>%
  ggplot(aes(x= alpha_e, y=alpha_l)) +
  geom_abline(aes(slope=1, intercept=0), lty=2, alpha=0.5) +
  geom_point()


loss_3 <- read_csv('~/projects/irt-position/Rsim/data3_loss.csv')

ggplot(loss_3, aes(x=epoch, y=loss)) +
  geom_line()


loss <- read_csv('~/projects/irt-position/Rsim/data2_experimental_loss.csv')

ggplot(loss, aes(x=epoch, y=loss)) +
  geom_line()
