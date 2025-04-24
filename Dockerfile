#Container for the Backend 

FROM 3.11
# current date to pass / yesterday
ARG curr
#region
ARG reg 

RUN cd /src
WORKDIR /src

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

RUN python3 CarbonCastE2E.py

RUN pip install --upgrade pip
COPY requirements.txt  /app/
RUN pip install --no-cache-dir -r requirements.txt

