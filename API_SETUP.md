# API Setup Instructions

## Setting Up Your Delta Exchange API Credentials

To use this trading bot, you need to configure your Delta Exchange API credentials in the `application.config` file.

### Step 1: Get API Credentials from Delta Exchange

1. Log in to your Delta Exchange account
2. Go to **Settings** → **API Management**
3. Create a new API key with the following permissions:
   - **Read**: Account information, positions, orders
   - **Trade**: Place and cancel orders
   - **Do NOT enable Withdraw** permissions for security

### Step 2: Configure application.config

Open the `application.config` file and add your credentials:

```ini
[API_CONFIGURATION]
api_key = YOUR_API_KEY_HERE
api_secret = YOUR_API_SECRET_HERE
paper_trading = true  # Set to false for live trading
```

### Step 3: Security Recommendations

- **NEVER commit your actual API credentials to git**
- Start with `paper_trading = true` to test the system
- Use API keys with minimal required permissions
- Regularly rotate your API keys
- Monitor your API usage and trading activity

### Step 4: Test Your Setup

Run the API connection test:
```bash
python test_delta_api.py
```

If successful, you'll see connection confirmation and account details.

### Step 5: Start Trading

Once your API is configured and tested:

```bash
# Start the web UI
python run_ui.py

# Or run directly
python delta_btc_strategy.py
```

## Important Notes

- The `application.config` file contains a complete template with all trading parameters
- Only the `api_key` and `api_secret` fields need to be filled in
- All other settings can be customized through the web UI or by editing the config file
- Always test with paper trading first before going live

## Troubleshooting

- **"Invalid API credentials"**: Double-check your API key and secret
- **"Permission denied"**: Ensure your API key has trading permissions enabled
- **"Rate limit exceeded"**: Wait a few minutes and try again

For more help, check the main documentation in `CLAUDE.md` or the Delta Exchange API docs.