library(tidyverse)

setwd('~/projects/irt-position')

d12 <- read_csv('Rsim-2/data1_loss.csv') %>%
  select(epoch, d1_2pl=loss)

d11 <- read_csv('Rsim-2/data1_1pl_loss.csv') %>%
  select(epoch, d1_1pl=loss)

d22 <- read_csv('Rsim-2/data2_loss.csv') %>%
  select(epoch, d2_2pl=loss)

d21 <- read_csv('Rsim-2/data2_1pl_loss.csv') %>%
  select(epoch, d2_1pl=loss)

d32 <- read_csv('Rsim-2/data3_loss.csv') %>%
  select(epoch, d3_2pl=loss)

d31 <- read_csv('Rsim-2/data3_1pl_loss.csv') %>%
  select(epoch, d3_1pl=loss)

d <- d11 %>%
  full_join(d12, by='epoch') %>%
  full_join(d21, by='epoch') %>%
  full_join(d22, by='epoch') %>%
  full_join(d31, by='epoch') %>%
  full_join(d32, by='epoch') %>%
  pivot_longer(-epoch, names_to='model', values_to='loss')

ggplot(d, aes(x=epoch, y=loss, color=model)) +
  geom_line() +
  theme_bw()
