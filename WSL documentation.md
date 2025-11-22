#  Why Need WSL

In order to integrate with vectorDB, if using windows, codes need to be run in WSL.

---

# Prerequisites

- Python in WSL2
- Code in WSL2
- MongoDB in Windows

# Connect local MongoDB in WSL

- Get windows host ip
```
export WIN_HOST_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
echo $WIN_HOST_IP
```

- Change the mongo url in **.env**
```
MONGO_URI = 'mongodb://localhost:27017'
```
->
```
MONGO_URI = 'mongodb://<your-windows-host-ip>:27017'
```

- If cannot connect MongoDB, edit mongod.cfg
```
Open C:\Program Files\MongoDB\Server\X.Y\bin\mongod.cfg
Change
    net:
        bindIp: 127.0.0.1
To
    net:
        bindIp: 127.0.0.1,<your-windows-host-ip>
Restart MongoDB on Win, run as administrator:
	net stop MongoDB
	net start MongoDB
```

- If still cannot connect MongoDB
```
Open Windows Defender Firewall > Advanced Settings
Inbound Rules → New Rule
Port → TCP → 27017 → Allow → Apply to all
Name it something like MongoDB_WSL
```

# Add proxy
```
export http_proxy="<proxy_address>:<port>"
export https_proxy="<proxy_address>:<port>"
export ftp_proxy="<proxy_address>:<port>"
export no_proxy="localhost,127.0.0.1"
```

# Install VectorDB
 - code in wsl2
 - docker in wsl2
 - weaviate in wsl2
 
##  How to run
-  streamlit run app2.py
