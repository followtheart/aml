# HTTPS 调用实验

服务端复用已有 AML FastAPI 接口，客户端仅需 Python 3 标准库。
启动参数使用 [Uvicorn 官方 HTTPS 配置](https://www.uvicorn.org/settings/#https)。
以下命令从仓库根目录执行，先安装 `memory_system/requirements.txt` 中的服务端依赖。

## 本机实验

生成仅用于实验的自签名证书（需要 OpenSSL）：

```bash
mkdir -p https_client/certs
chmod 700 https_client/certs
openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 30 \
  -keyout https_client/certs/server.key -out https_client/certs/server.crt \
  -subj '/CN=localhost' -addext 'subjectAltName=DNS:localhost,IP:127.0.0.1'
chmod 600 https_client/certs/server.key
```

在终端一启动服务（默认监听 `0.0.0.0:8443`，Ctrl+C 停止）：

```bash
export AML_API_KEY='replace-with-your-random-token'
AML_FAKE=1 AML_DB_PATH="$PWD/https_client/experiment.db" \
SSL_CERTFILE="$PWD/https_client/certs/server.crt" \
SSL_KEYFILE="$PWD/https_client/certs/server.key" \
bash memory_system/scripts/start_https.sh
```

在终端二运行客户端，使用与服务端相同的 token：

```bash
export AML_API_KEY='replace-with-your-random-token'
python3 https_client/client.py --ca https_client/certs/server.crt
```

依次调用 `/health`、`/add`、`/search`，输出 JSON 与 PASS；失败时非零退出。
每次创建独立实验用户，数据保留在实验数据库中。`AML_FAKE=1` 不调用模型，
用于验证 TLS、鉴权和接口链路，不代表真实模型效果。
可用 `--text`、`--query`、`--user-id` 自定义实验，`--health-only` 只检查连接。
客户端始终验证证书和主机名，不自动跟随重定向。

## 外部机器调用

1. 将证书换成包含实际域名或 IP 的证书。自签名证书的 `subjectAltName`
   必须包含客户端访问的域名（`DNS:...`）或 IP（`IP:...`）。
2. 服务端设置 `SSL_CERTFILE` 为证书链、`SSL_KEYFILE` 为私钥；可用 `PORT` 修改端口。
   按部署环境放行 TCP 8443，包括主机防火墙、云安全组和必要的 NAT 端口映射。
3. 客户端运行 `python3 https_client/client.py --url https://实际域名:8443`。
   自签名证书需将**证书文件**复制到客户端，并添加 `--ca /path/to/server.crt`；
   私钥只留在服务端。公信 CA 证书通常不需要 `--ca`。

实际模型实验移除 `AML_FAKE=1`，沿用项目已有的模型配置。
脚本继承环境变量，不自动读取 `.env`；`AML_API_KEY` 是接口鉴权 token，
与模型供应商的 API key 不同。`/health` 依照现有接口保持无需鉴权。
