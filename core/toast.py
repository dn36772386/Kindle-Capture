# -*- coding: utf-8 -*-
import subprocess, shutil, sys

def show_toast(title: str, message: str):
    """ winrtがある場合はそれを使い、無い場合はPowerShellにフォールバック。 """
    try:
        import winrt.windows.ui.notifications as wnot
        import winrt.windows.data.xml.dom as wdom

        # シンプルなトースト（アプリID不要の簡易版）
        # 注意: 本格運用ではAUMID設定が望ましい
        t = f"""
        <toast>
          <visual>
            <binding template="ToastGeneric">
              <text>{title}</text>
              <text>{message}</text>
            </binding>
          </visual>
        </toast>
        """
        xml = wdom.XmlDocument()
        xml.load_xml(t)
        n = wnot.ToastNotification(xml)
        mgr = wnot.ToastNotificationManager.create_toast_notifier("SnapLite")
        mgr.show(n)
        return
    except Exception:
        pass

    # PowerShellフォールバック（BurntToast未インストールでも簡易表示に挑戦）
    try:
        ps = shutil.which("powershell") or "powershell"
        cmd = f'''[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;
$tmpl = @"
<toast><visual><binding template="ToastGeneric">
<text>{title}</text><text>{message}</text>
</binding></visual></toast>
"@;
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument;
$xml.LoadXml($tmpl);
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml);
$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("SnapLite");
$notifier.Show($toast);'''
        subprocess.Popen([ps, "-NoProfile", "-Command", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # 何もできない場合は黙る
        pass
