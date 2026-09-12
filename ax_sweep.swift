// ax_sweep.swift — dump every running app's accessibility nodes whose text
// matches call keywords. Diagnostic for "where does the incoming-call UI
// live on this macOS?". Usage: swift ax_sweep.swift [seconds] > out.jsonl
// Samples once per second for N seconds (default 60); prints one JSON line
// per NEW (process, role, text) node seen. Needs Accessibility trust.
import AppKit
import ApplicationServices
import Foundation

let keywords = ["accept", "answer", "decline", "facetime", "incoming", "audio", "spencer", "call"]
let seconds = Int(CommandLine.arguments.dropFirst().first ?? "60") ?? 60
var seen = Set<String>()

func attr(_ el: AXUIElement, _ name: String) -> Any? {
    var v: CFTypeRef?
    return AXUIElementCopyAttributeValue(el, name as CFString, &v) == .success ? v : nil
}
func str(_ el: AXUIElement, _ name: String) -> String {
    (attr(el, name) as? String) ?? ""
}
func walk(_ el: AXUIElement, _ proc: String, _ depth: Int, _ out: inout [[String: Any]]) {
    if depth > 14 { return }
    let role = str(el, kAXRoleAttribute)
    let texts = [str(el, kAXTitleAttribute), str(el, kAXDescriptionAttribute), str(el, kAXValueAttribute), str(el, kAXHelpAttribute), str(el, kAXIdentifierAttribute)].filter { !$0.isEmpty }
    let joined = texts.joined(separator: " | ").lowercased()
    if keywords.contains(where: { joined.contains($0) }) {
        var actions: CFArray?
        AXUIElementCopyActionNames(el, &actions)
        var enabled: CFTypeRef?
        AXUIElementCopyAttributeValue(el, kAXEnabledAttribute as CFString, &enabled)
        out.append(["process": proc, "role": role, "subrole": str(el, kAXSubroleAttribute), "texts": texts,
                    "actions": (actions as? [String]) ?? [], "enabled": (enabled as? Bool) ?? false, "depth": depth])
    }
    if let children = attr(el, kAXChildrenAttribute) as? [AXUIElement] {
        for c in children.prefix(200) { walk(c, proc, depth + 1, &out) }
    }
}

guard AXIsProcessTrusted() else { FileHandle.standardError.write("not trusted\n".data(using: .utf8)!); exit(1) }
let start = Date()
while Date().timeIntervalSince(start) < Double(seconds) {
    for app in NSWorkspace.shared.runningApplications {
        guard app.activationPolicy != .prohibited || (app.localizedName ?? "").lowercased().contains("notification") || (app.bundleIdentifier ?? "").lowercased().contains("facetime") || (app.bundleIdentifier ?? "").lowercased().contains("phone") || (app.bundleIdentifier ?? "").lowercased().contains("telephony") || (app.bundleIdentifier ?? "").lowercased().contains("call") else { continue }
        let el = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(el, 0.5)
        var out: [[String: Any]] = []
        let proc = "\(app.localizedName ?? "?") [\(app.bundleIdentifier ?? "?")]"
        walk(el, proc, 0, &out)
        for node in out {
            let key = "\(node["process"]!)|\(node["role"]!)|\(node["texts"]!)"
            if seen.insert(key).inserted, let data = try? JSONSerialization.data(withJSONObject: node) {
                let stamp = ISO8601DateFormatter().string(from: Date())
                print("\(stamp) \(String(decoding: data, as: UTF8.self))")
                fflush(stdout)
            }
        }
    }
    Thread.sleep(forTimeInterval: 1.0)
}
