
ARG IMAGE
FROM $IMAGE

RUN git clone https://github.com/supervisely-ecosystem/yolov5.git -b embed-tracking-to-serve-app

RUN pip install -r /yolov5/requirements.txt

RUN apt-get update && apt-get install -y openssh-server
EXPOSE 22

RUN apt-get install -y sudo
RUN mkdir -p /run/sshd



# MORE SECURE: use a trusted RSA key
ARG home=/root
RUN mkdir $home/.ssh
COPY authorized_keys $home/.ssh/authorized_keys
RUN chown root:root $home/.ssh/authorized_keys && \
    chmod 600 $home/.ssh/authorized_keys

COPY sshd_deamon.sh /sshd_deamon.sh
RUN chmod 755 /sshd_deamon.sh
CMD ["/sshd_deamon.sh"]
ENTRYPOINT ["sh", "-c", "/sshd_deamon.sh"]