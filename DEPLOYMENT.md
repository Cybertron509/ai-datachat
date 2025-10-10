# AI DataChat - Deployment Guide

**Copyright © 2025 Gardel Hiram. All rights reserved.**

This guide provides instructions for deploying AI DataChat. Note that deployment of this software requires explicit permission from the copyright holder.

---

## ⚖️ Legal Notice

Before deploying AI DataChat, ensure you have:
1. ✅ Explicit written permission from Gardel Hiram
2. ✅ A valid commercial license (if applicable)
3. ✅ Signed deployment agreement

Unauthorized deployment is strictly prohibited and may result in legal action.

Contact: gardelhiram9@gmail.com

---

## 🚀 Deployment Options

### Option 1: Render.com (Current Production)

**Live Demo**: https://ai-datachat.onrender.com

1. **Create Render Account**
   - Sign up at https://render.com

2. **Connect Repository**
   - Link your GitHub repository
   - Select the ai-datachat repository

3. **Configure Web Service**
   ```yaml
   Name: ai-datachat
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

4. **Set Environment Variables**
   ```
   OPENAI_API_KEY=your_key
   STRIPE_SECRET_KEY=your_key
   STRIPE_PUBLISHABLE_KEY=your_key
   STRIPE_PRO_PRICE_ID=your_price_id
   SECRET_KEY=your_secret
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete

---

### Option 2: Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit https://streamlit.io/cloud

2. **Deploy New App**
   - Click "New app"
   - Select repository: Cybertron509/ai-datachat
   - Main file: app.py

3. **Configure Secrets**
   
   Go to App Settings → Secrets and add:
   ```toml
   [secrets]
   OPENAI_API_KEY = "your_key"
   STRIPE_SECRET_KEY = "your_key"
   STRIPE_PUBLISHABLE_KEY = "your_key"
   STRIPE_PRO_PRICE_ID = "your_price_id"
   SECRET_KEY = "your_secret"
   ```

4. **Deploy**

---

### Option 3: Heroku

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   
   # Ubuntu
   sudo snap install --classic heroku
   ```

2. **Login and Create App**
   ```bash
   heroku login
   heroku create ai-datachat-prod
   ```

3. **Add Buildpacks**
   ```bash
   heroku buildpacks:set heroku/python
   ```

4. **Create Procfile**
   ```
   web: sh setup.sh && streamlit run app.py
   ```

5. **Create setup.sh**
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

6. **Set Environment Variables**
   ```bash
   heroku config:set OPENAI_API_KEY=your_key
   heroku config:set STRIPE_SECRET_KEY=your_key
   heroku config:set SECRET_KEY=your_secret
   ```

7. **Deploy**
   ```bash
   git push heroku main
   ```

---

### Option 4: Docker

1. **Create Dockerfile**
   ```dockerfile
   # Copyright © 2025 Gardel Hiram. All rights reserved.
   
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Create .dockerignore**
   ```
   __pycache__
   *.pyc
   *.pyo
   .env
   .git
   venv
   .venv
   ```

3. **Build and Run**
   ```bash
   docker build -t ai-datachat .
   docker run -p 8501:8501 --env-file .env ai-datachat
   ```

---

### Option 5: AWS EC2

1. **Launch EC2 Instance**
   - AMI: Ubuntu Server 22.04 LTS
   - Instance Type: t2.medium (minimum)
   - Security Group: Allow ports 22, 80, 443, 8501

2. **Connect and Setup**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Python
   sudo apt install python3-pip python3-venv -y
   
   # Clone repository (requires permission)
   git clone https://github.com/Cybertron509/ai-datachat.git
   cd ai-datachat
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   nano .env
   # Add your environment variables
   ```

4. **Run with Supervisor**
   ```bash
   sudo apt install supervisor -y
   sudo nano /etc/supervisor/conf.d/ai-datachat.conf
   ```
   
   Add:
   ```ini
   [program:ai-datachat]
   directory=/home/ubuntu/ai-datachat
   command=/home/ubuntu/ai-datachat/venv/bin/streamlit run app.py --server.port=8501
   autostart=true
   autorestart=true
   user=ubuntu
   ```

5. **Start Service**
   ```bash
   sudo supervisorctl reread
   sudo supervisorctl update
   sudo supervisorctl start ai-datachat
   ```

---

## 🔐 Security Considerations

### SSL/TLS Configuration
- Always use HTTPS in production
- Obtain SSL certificates from Let's Encrypt or your provider
- Configure your reverse proxy (nginx/Apache) for SSL

### Environment Variables
- Never commit `.env` files to version control
- Use secure secret management (AWS Secrets Manager, etc.)
- Rotate API keys regularly

### Database Security
- Use encrypted connections
- Implement proper backup strategies
- Restrict database access

### Application Security
- Keep dependencies updated
- Monitor for security vulnerabilities
- Implement rate limiting
- Use WAF (Web Application Firewall)

---

## 📊 Monitoring

### Recommended Tools
- **Uptime Monitoring**: UptimeRobot, Pingdom
- **Application Monitoring**: New Relic, Datadog
- **Error Tracking**: Sentry
- **Logs**: CloudWatch, Papertrail

---

## 🔄 CI/CD Pipeline

### GitHub Actions Example

```yaml
# .github/workflows/deploy.yml
# Copyright © 2025 Gardel Hiram. All rights reserved.

name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to Render
        env:
          RENDER_API_KEY: ${{ secrets.RENDER_API_KEY }}
        run: |
          curl -X POST "https://api.render.com/v1/services/$SERVICE_ID/deploys" \
            -H "Authorization: Bearer $RENDER_API_KEY"
```

---

## 📝 Post-Deployment Checklist

- [ ] Verify all environment variables are set
- [ ] Test authentication system
- [ ] Verify file upload functionality
- [ ] Test AI chat features
- [ ] Verify payment processing (if enabled)
- [ ] Check all visualizations render correctly
- [ ] Test export features
- [ ] Verify SSL certificate
- [ ] Set up monitoring and alerts
- [ ] Configure backup systems
- [ ] Test error handling
- [ ] Verify mobile responsiveness
- [ ] Check performance metrics
- [ ] Review security headers
- [ ] Test all Pro features

---

## 🆘 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find process using port 8501
lsof -i :8501
# Kill the process
kill -9 <PID>
```

**Memory Issues**
- Increase instance RAM
- Optimize data processing
- Implement data pagination

**Slow Performance**
- Enable caching
- Optimize database queries
- Use CDN for static assets
- Implement load balancing

---

## 📞 Support

For deployment support or licensing inquiries:

**Gardel Hiram**
- Email: gardelhiram9@gmail.com
- GitHub: @Cybertron509

---

## ⚖️ Legal Disclaimer

This deployment guide is provided for authorized users only. Unauthorized deployment
of AI DataChat constitutes copyright infringement and violation of intellectual
property rights.

**All deployment activities must comply with the terms specified in the LICENSE file.**

---

**Copyright © 2025 Gardel Hiram. All rights reserved.**

*Last Updated: October 09, 2025*
