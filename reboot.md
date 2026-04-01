# Internet Monitor Test

## 1. Set timer to 40s (test mode)
```bash
sudo sed -i 's|OnBootSec=100s|OnBootSec=40s|; s|OnUnitActiveSec=100s|OnUnitActiveSec=40s|' /etc/systemd/system/internet-monitor.timer && sudo systemctl daemon-reload && sudo systemctl restart internet-monitor.timer
```

## 2. Block connectivity to trigger failures
```bash
sudo iptables -A OUTPUT -p icmp -j DROP && sudo iptables -A OUTPUT -p tcp --dport 80 -j DROP && sudo iptables -A OUTPUT -p tcp --dport 443 -j DROP
```

## 3. Watch the log (expect 3 failures ~40s apart, then 60s grace, then reboot)
```bash
tail -f /var/log/internet-monitor.log
```

---

# Post-Reboot Cleanup

## 4. Clear iptables rules (in case they persist)
```bash
sudo iptables -D OUTPUT -p icmp -j DROP 2>/dev/null; sudo iptables -D OUTPUT -p tcp --dport 80 -j DROP 2>/dev/null; sudo iptables -D OUTPUT -p tcp --dport 443 -j DROP 2>/dev/null
```

## 5. Restore timer to 100s (production setting)
```bash
sudo sed -i 's|OnBootSec=40s|OnBootSec=100s|; s|OnUnitActiveSec=40s|OnUnitActiveSec=100s|' /etc/systemd/system/internet-monitor.timer && sudo systemctl daemon-reload && sudo systemctl restart internet-monitor.timer
```

## 6. Verify everything is back to normal
```bash
systemctl list-timers | grep internet
tail -20 /var/log/internet-monitor.log
```
