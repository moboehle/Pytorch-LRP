FROM jupyter/scipy-notebook:76402a27fd13

WORKDIR /home/jovyan/work

USER root

# Style profile
RUN echo "export CLICOLOR=1" >> /root/.bashrc \
  && echo "export PS1='\u@\h:\[\e[0;34m\]\W\[\e[0m\]\$ '" >> /root/.bashrc \
  && echo "export LS_OPTIONS='--color=auto'" >> /root/.bashrc \
  && echo "export LSCOLORS=excxhxDxfxhxhxhxhxFxFx" >> /root/.bashrc \
  && echo 'alias ls="ls $LS_OPTIONS"' >> /root/.bashrc \
  && echo 'alias ll="ls $LS_OPTIONS -l"' >> /root/.bashrc \
  && echo 'alias l="ls $LS_OPTIONS -lA"' >> /root/.bashrc

USER jovyan

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
