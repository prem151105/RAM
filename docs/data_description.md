# Data Description

## Overview

The AgentNet Customer Support System uses two main types of data:
1. Historical ticket data
2. Customer conversation data

## Data Sources

### 1. Historical Tickets (`tickets.csv`)

This dataset contains resolved customer support tickets with the following structure:

```csv
ticket_id,issue_type,description,resolution,resolution_time,created_at,resolved_at
```

Fields:
- `ticket_id`: Unique identifier for the ticket
- `issue_type`: Category of the issue (e.g., account, device, network)
- `description`: Detailed description of the issue
- `resolution`: How the issue was resolved
- `resolution_time`: Time taken to resolve (in minutes)
- `created_at`: Timestamp when ticket was created
- `resolved_at`: Timestamp when ticket was resolved

### 2. Customer Conversations

The system uses five types of conversation datasets:

1. **Account Sync Issues** (`conv_account_sync.csv`)
   - Format: `conversation_id,message,timestamp,agent_id`
   - Content: Conversations about account synchronization problems

2. **Device Compatibility** (`conv_device_compat.csv`)
   - Format: `conversation_id,message,timestamp,agent_id`
   - Content: Conversations about device compatibility issues

3. **Network Issues** (`conv_network_issue.csv`)
   - Format: `conversation_id,message,timestamp,agent_id`
   - Content: Conversations about network connectivity problems

4. **Payment Gateway** (`conv_payment_gateway.csv`)
   - Format: `conversation_id,message,timestamp,agent_id`
   - Content: Conversations about payment processing issues

5. **Software Installation** (`conv_software_install.csv`)
   - Format: `conversation_id,message,timestamp,agent_id`
   - Content: Conversations about software installation problems

## Data Processing

### Raw Data
- Located in `data/raw/`
- Original, unprocessed datasets
- Maintained for reproducibility

### Processed Data
- Located in `data/processed/`
- Cleaned and formatted for system use
- Includes:
  - Tokenized conversations
  - Embeddings for RAG
  - Normalized timestamps
  - Categorized issues

## Data Quality

The system implements several data quality checks:
- Missing value handling
- Duplicate detection
- Timestamp validation
- Text cleaning and normalization
- Category consistency verification

## Data Privacy

All customer data is handled with strict privacy measures:
- Personal information is anonymized
- Sensitive data is encrypted
- Access is restricted to authorized personnel
- Data retention policies are enforced 