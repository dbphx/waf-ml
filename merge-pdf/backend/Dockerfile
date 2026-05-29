FROM golang:1.25-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /merge-pdf-server ./cmd/server

FROM alpine:3.21

RUN adduser -D -u 10001 appuser

WORKDIR /app

COPY --from=builder /merge-pdf-server /usr/local/bin/merge-pdf-server

USER appuser

EXPOSE 8080

ENTRYPOINT ["merge-pdf-server"]

