FROM python:3.8-buster as base

# apt-get and system utilities
RUN apt-get update \
    && apt-get install -y curl apt-transport-https debconf-utils vim unixodbc-dev locales \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/9/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 mssql-tools

# install SQL Server drivers and tools
RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
RUN locale-gen en_US.UTF-8
RUN update-locale

COPY requirements.txt ./
RUN pip install -r requirements.txt --upgrade

WORKDIR /code
ENTRYPOINT "/bin/bash"