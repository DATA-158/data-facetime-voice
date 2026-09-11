// audio_procs.swift — which processes have which audio devices open (macOS 14+).
// Usage: swift audio_procs.swift [filter-substring]
import CoreAudio
import Foundation

func getData<T>(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector, _ type: T.Type) -> [T] {
    var addr = AudioObjectPropertyAddress(mSelector: sel, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(obj, &addr, 0, nil, &size) == noErr, size > 0 else { return [] }
    let count = Int(size) / MemoryLayout<T>.stride
    var buf = [T](unsafeUninitializedCapacity: count) { p, n in n = count }
    guard AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, &buf) == noErr else { return [] }
    return buf
}
func getString(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector) -> String {
    var addr = AudioObjectPropertyAddress(mSelector: sel, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(MemoryLayout<CFString?>.size)
    var s: Unmanaged<CFString>?
    guard AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, &s) == noErr, let s else { return "?" }
    return s.takeRetainedValue() as String
}
func getU32(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector) -> UInt32 { getData(obj, sel, UInt32.self).first ?? 0 }
func getPid(_ obj: AudioObjectID) -> pid_t { getData(obj, kAudioProcessPropertyPID, pid_t.self).first ?? -1 }

let filter = CommandLine.arguments.dropFirst().first?.lowercased()
let procs = getData(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyProcessObjectList, AudioObjectID.self)
var shown = 0
for p in procs {
    let bundle = getString(p, kAudioProcessPropertyBundleID)
    let pid = getPid(p)
    let name = { () -> String in
        if let app = NSRunningApplication(processIdentifier: pid) { return app.localizedName ?? bundle }
        return bundle
    }()
    let running = getU32(p, kAudioProcessPropertyIsRunning) != 0
    let runIn = getU32(p, kAudioProcessPropertyIsRunningInput) != 0
    let runOut = getU32(p, kAudioProcessPropertyIsRunningOutput) != 0
    let devs = getData(p, kAudioProcessPropertyDevices, AudioObjectID.self).map { getString($0, kAudioObjectPropertyName) }
    if let f = filter, !(name.lowercased().contains(f) || bundle.lowercased().contains(f)) { continue }
    if !running && devs.isEmpty && filter == nil { continue }
    print("pid=\(pid) \(name) [\(bundle)] running=\(running) input=\(runIn) output=\(runOut) devices=\(devs)")
    shown += 1
}
if shown == 0 { print("(no matching audio processes)") }
