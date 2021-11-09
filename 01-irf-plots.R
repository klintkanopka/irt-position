library(tidyverse)

setwd('~/projects/irt-position')

irf <- function(theta, k, c, s, b0, b1, a0=1, a1=1){
  p0 <- plogis(a0*(theta - b0))
  p1 <- plogis(a1*(theta - b1))
  pi <- plogis(c*(k-s))
  p <- pi*p0 + (1-pi)*p1
  return(p)
} 

mix_param <- function(k, c, s){
  pi <- plogis(c*(k-s))
  return(pi)
}

th_grid <- seq(-5, 5, by=0.01)

data.frame(
  theta = rep(th_grid, 8),
  s = rep(c(0, 0.33, 0.66, 1), each=length(th_grid), times=2),
  k = rep(c(0.25, 0.75), each=4*length(th_grid)),
  b0 = -1,
  b1 = 1,
  c = 7) %>% 
  mutate(p = irf(theta, k, c, s, b0, b1)) %>% 
  ggplot(aes(x = theta, y=p, lty=as.factor(k), color=as.factor(s))) +
  geom_line() +
  #facet_grid(.~as.factor(k)) +
  theme_bw()

ggsave('fig/irf_one_panel.png', width=8, height=6)


data.frame(
  theta = rep(th_grid, 8),
  s = rep(c(0, 0.33, 0.66, 1), each=length(th_grid), times=2),
  k = rep(c(0.25, 0.75), each=4*length(th_grid)),
  b0 = -1,
  b1 = 1,
  c = 7) %>% 
  mutate(p = irf(theta, k, c, s, b0, b1)) %>% 
  ggplot(aes(x = theta, y=p, color=as.factor(s))) +
  geom_line() +
  facet_grid(.~as.factor(k)) +
  theme_bw()

ggsave('fig/irf_two_panel.png', width=8, height=6)

pos_grid <- seq(0, 1, by=0.01)

data.frame(
  s = rep(pos_grid, 5),
  k = rep(c(0,0.25,0.5,0.75,1), each=length(pos_grid)),
  c = 7) %>%
  mutate(pi = mix_param(k, c, s)) %>% 
  ggplot(aes(x = s, y=pi, color=as.factor(k))) +
  geom_line()

ggsave('fig/mixing_param.png', width=8, height=6)

  