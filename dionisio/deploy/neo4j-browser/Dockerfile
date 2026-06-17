FROM neo4j:5-community

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY index.html /opt/dionisio/browser/index.html
COPY proxy.py /opt/dionisio/browser/proxy.py
COPY docker-entrypoint-dionisio.sh /startup/docker-entrypoint-dionisio.sh

RUN chmod +x /startup/docker-entrypoint-dionisio.sh \
    && chown -R neo4j:neo4j /opt/dionisio

ENTRYPOINT ["/startup/docker-entrypoint-dionisio.sh"]
CMD ["neo4j"]
