uint8 HEARTBEAT = 0
uint8 COORDINATION = 1

uint8 msg_type
string sender_id
string recipient_id           # Empty string = broadcast
uint64 timestamp
bool overwrite_targeted       # If broadcast, whether to overwrite recipient-specific messages
uint8[] payload               # Serialized inner message
