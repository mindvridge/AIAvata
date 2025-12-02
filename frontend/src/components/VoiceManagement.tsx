/**
 * Voice Management Component
 *
 * Zonos TTS 음성 프로필 관리 페이지
 * - 음성 목록 조회
 * - 새 음성 복제 (오디오 업로드)
 * - 음성 편집/삭제
 * - 음성 미리듣기
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';

// API base URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Types
interface VoiceProfile {
  id: string;
  name: string;
  description: string;
  language: string;
  duration_seconds: number;
  sample_rate: number;
  created_at: string;
  updated_at: string;
}

interface VoiceListResponse {
  voices: VoiceProfile[];
  total: number;
  supported_languages: string[];
}

// Language mapping
const LANGUAGE_NAMES: Record<string, string> = {
  ko: '한국어',
  en: 'English',
  ja: '日本語',
  zh: '中文',
  fr: 'Français',
  de: 'Deutsch',
};

// Emotion options
const EMOTIONS = [
  'neutral',
  'happy',
  'sad',
  'angry',
  'fear',
  'surprise',
  'disgust',
];

interface VoiceManagementProps {
  onClose?: () => void;
  onSelectVoice?: (voiceId: string) => void;
}

export function VoiceManagement({ onClose, onSelectVoice }: VoiceManagementProps) {
  // State
  const [voices, setVoices] = useState<VoiceProfile[]>([]);
  const [supportedLanguages, setSupportedLanguages] = useState<string[]>(['ko', 'en', 'ja', 'zh', 'fr', 'de']);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Create voice state
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [createName, setCreateName] = useState('');
  const [createDescription, setCreateDescription] = useState('');
  const [createLanguage, setCreateLanguage] = useState('ko');
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [creating, setCreating] = useState(false);

  // Edit state
  const [editingVoice, setEditingVoice] = useState<VoiceProfile | null>(null);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editLanguage, setEditLanguage] = useState('en');

  // Preview state
  const [previewVoiceId, setPreviewVoiceId] = useState<string | null>(null);
  const [previewText, setPreviewText] = useState('Hello, this is a voice preview test.');
  const [previewEmotion, setPreviewEmotion] = useState('neutral');
  const [previewRate, setPreviewRate] = useState(1.0);
  const [previewing, setPreviewing] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  // File input ref
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch voices
  const fetchVoices = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/voices`);
      if (!response.ok) throw new Error('Failed to fetch voices');

      const data: VoiceListResponse = await response.json();
      setVoices(data.voices);
      setSupportedLanguages(data.supported_languages);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchVoices();
  }, [fetchVoices]);

  // Create voice
  const handleCreate = async () => {
    if (!audioFile || !createName.trim()) {
      setError('Name and audio file are required');
      return;
    }

    try {
      setCreating(true);
      const formData = new FormData();
      formData.append('audio', audioFile);
      formData.append('name', createName.trim());
      formData.append('description', createDescription.trim());
      formData.append('language', createLanguage);

      const response = await fetch(`${API_BASE_URL}/api/voices`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to create voice');
      }

      await fetchVoices();
      setShowCreateForm(false);
      setCreateName('');
      setCreateDescription('');
      setCreateLanguage('en');
      setAudioFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create voice');
    } finally {
      setCreating(false);
    }
  };

  // Update voice
  const handleUpdate = async () => {
    if (!editingVoice) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/voices/${editingVoice.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: editName.trim() || null,
          description: editDescription.trim() || null,
          language: editLanguage || null,
        }),
      });

      if (!response.ok) throw new Error('Failed to update voice');

      await fetchVoices();
      setEditingVoice(null);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update voice');
    }
  };

  // Delete voice
  const handleDelete = async (voiceId: string) => {
    if (!confirm('Are you sure you want to delete this voice?')) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/voices/${voiceId}`, {
        method: 'DELETE',
      });

      if (!response.ok) throw new Error('Failed to delete voice');

      await fetchVoices();
      setError(null);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete voice');
    }
  };

  // Preview voice
  const handlePreview = async (voiceId: string) => {
    try {
      setPreviewing(true);
      setPreviewVoiceId(voiceId);

      const response = await fetch(`${API_BASE_URL}/api/voices/${voiceId}/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: previewText,
          voice_id: voiceId,
          language: voices.find(v => v.id === voiceId)?.language || 'en',
          emotion: previewEmotion,
          speaking_rate: previewRate,
        }),
      });

      if (!response.ok) throw new Error('Failed to preview voice');

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);

      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.play();
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to preview voice');
    } finally {
      setPreviewing(false);
    }
  };

  // Download original audio
  const handleDownload = async (voiceId: string, name: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/voices/${voiceId}/audio`);
      if (!response.ok) throw new Error('Failed to download audio');

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${name}.wav`;
      a.click();
      URL.revokeObjectURL(url);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download audio');
    }
  };

  // Start editing
  const startEdit = (voice: VoiceProfile) => {
    setEditingVoice(voice);
    setEditName(voice.name);
    setEditDescription(voice.description);
    setEditLanguage(voice.language);
  };

  // Format duration
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return mins > 0 ? `${mins}:${secs.toString().padStart(2, '0')}` : `${secs}s`;
  };

  // Format date
  const formatDate = (dateStr: string) => {
    try {
      return new Date(dateStr).toLocaleDateString();
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-xl w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <div>
            <h2 className="text-xl font-bold text-white">Voice Management</h2>
            <p className="text-sm text-gray-400">Zonos TTS - Voice Cloning & Management</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Error message */}
        {error && (
          <div className="mx-4 mt-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
            {error}
            <button
              onClick={() => setError(null)}
              className="ml-2 text-red-300 hover:text-white"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {/* Create button */}
          {!showCreateForm && (
            <button
              onClick={() => setShowCreateForm(true)}
              className="mb-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Clone New Voice
            </button>
          )}

          {/* Create form */}
          {showCreateForm && (
            <div className="mb-6 p-4 bg-gray-800 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">Clone New Voice</h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Name *</label>
                  <input
                    type="text"
                    value={createName}
                    onChange={(e) => setCreateName(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                    placeholder="Voice name"
                  />
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-1">Language</label>
                  <select
                    value={createLanguage}
                    onChange={(e) => setCreateLanguage(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                  >
                    {supportedLanguages.map((lang) => (
                      <option key={lang} value={lang}>
                        {LANGUAGE_NAMES[lang] || lang}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="md:col-span-2">
                  <label className="block text-sm text-gray-400 mb-1">Description</label>
                  <input
                    type="text"
                    value={createDescription}
                    onChange={(e) => setCreateDescription(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                    placeholder="Optional description"
                  />
                </div>

                <div className="md:col-span-2">
                  <label className="block text-sm text-gray-400 mb-1">
                    Reference Audio * (10-30 seconds recommended)
                  </label>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*"
                    onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
                    className="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 file:mr-4 file:py-1 file:px-3 file:rounded file:border-0 file:bg-blue-600 file:text-white file:cursor-pointer"
                  />
                  {audioFile && (
                    <p className="mt-1 text-sm text-gray-400">
                      Selected: {audioFile.name} ({(audioFile.size / 1024 / 1024).toFixed(2)} MB)
                    </p>
                  )}
                </div>
              </div>

              <div className="flex gap-2 mt-4">
                <button
                  onClick={handleCreate}
                  disabled={creating || !audioFile || !createName.trim()}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg flex items-center gap-2"
                >
                  {creating ? (
                    <>
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Creating...
                    </>
                  ) : (
                    'Create Voice'
                  )}
                </button>
                <button
                  onClick={() => {
                    setShowCreateForm(false);
                    setAudioFile(null);
                    if (fileInputRef.current) fileInputRef.current.value = '';
                  }}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded-lg"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Preview controls */}
          <div className="mb-4 p-3 bg-gray-800 rounded-lg">
            <div className="flex flex-wrap gap-4 items-end">
              <div className="flex-1 min-w-[200px]">
                <label className="block text-sm text-gray-400 mb-1">Preview Text</label>
                <input
                  type="text"
                  value={previewText}
                  onChange={(e) => setPreviewText(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none text-sm"
                  placeholder="Text to synthesize"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Emotion</label>
                <select
                  value={previewEmotion}
                  onChange={(e) => setPreviewEmotion(e.target.value)}
                  className="px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none text-sm"
                >
                  {EMOTIONS.map((emotion) => (
                    <option key={emotion} value={emotion}>
                      {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Rate: {previewRate.toFixed(1)}x</label>
                <input
                  type="range"
                  min="0.5"
                  max="2"
                  step="0.1"
                  value={previewRate}
                  onChange={(e) => setPreviewRate(parseFloat(e.target.value))}
                  className="w-24"
                />
              </div>
            </div>
          </div>

          {/* Voice list */}
          {loading ? (
            <div className="text-center py-8">
              <svg className="w-8 h-8 animate-spin mx-auto text-blue-500" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <p className="mt-2 text-gray-400">Loading voices...</p>
            </div>
          ) : voices.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <svg className="w-12 h-12 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
              <p>No voice profiles yet.</p>
              <p className="text-sm mt-1">Click "Clone New Voice" to create one.</p>
            </div>
          ) : (
            <div className="grid gap-3">
              {voices.map((voice) => (
                <div
                  key={voice.id}
                  className="p-4 bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors"
                >
                  {editingVoice?.id === voice.id ? (
                    /* Edit mode */
                    <div className="space-y-3">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <input
                          type="text"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          className="px-3 py-2 bg-gray-700 rounded-lg border border-gray-600"
                          placeholder="Name"
                        />
                        <input
                          type="text"
                          value={editDescription}
                          onChange={(e) => setEditDescription(e.target.value)}
                          className="px-3 py-2 bg-gray-700 rounded-lg border border-gray-600"
                          placeholder="Description"
                        />
                        <select
                          value={editLanguage}
                          onChange={(e) => setEditLanguage(e.target.value)}
                          className="px-3 py-2 bg-gray-700 rounded-lg border border-gray-600"
                        >
                          {supportedLanguages.map((lang) => (
                            <option key={lang} value={lang}>
                              {LANGUAGE_NAMES[lang] || lang}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={handleUpdate}
                          className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm"
                        >
                          Save
                        </button>
                        <button
                          onClick={() => setEditingVoice(null)}
                          className="px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded text-sm"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    /* View mode */
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <h4 className="font-semibold">{voice.name}</h4>
                          <span className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-300">
                            {LANGUAGE_NAMES[voice.language] || voice.language}
                          </span>
                          <span className="text-xs text-gray-500">
                            {formatDuration(voice.duration_seconds)}
                          </span>
                        </div>
                        {voice.description && (
                          <p className="text-sm text-gray-400 mt-1">{voice.description}</p>
                        )}
                        <p className="text-xs text-gray-500 mt-1">
                          Created: {formatDate(voice.created_at)} | ID: {voice.id}
                        </p>
                      </div>

                      <div className="flex items-center gap-2 ml-4">
                        {/* Preview button */}
                        <button
                          onClick={() => handlePreview(voice.id)}
                          disabled={previewing && previewVoiceId === voice.id}
                          className="p-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg transition-colors"
                          title="Preview"
                        >
                          {previewing && previewVoiceId === voice.id ? (
                            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                          ) : (
                            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                              <path d="M8 5v14l11-7z" />
                            </svg>
                          )}
                        </button>

                        {/* Select button */}
                        {onSelectVoice && (
                          <button
                            onClick={() => onSelectVoice(voice.id)}
                            className="p-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                            title="Select this voice"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          </button>
                        )}

                        {/* Download button */}
                        <button
                          onClick={() => handleDownload(voice.id, voice.name)}
                          className="p-2 bg-gray-600 hover:bg-gray-500 rounded-lg transition-colors"
                          title="Download original audio"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                          </svg>
                        </button>

                        {/* Edit button */}
                        <button
                          onClick={() => startEdit(voice)}
                          className="p-2 bg-gray-600 hover:bg-gray-500 rounded-lg transition-colors"
                          title="Edit"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                          </svg>
                        </button>

                        {/* Delete button */}
                        <button
                          onClick={() => handleDelete(voice.id)}
                          className="p-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                          title="Delete"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-700 bg-gray-800/50">
          <div className="flex items-center justify-between text-sm text-gray-400">
            <span>
              {voices.length} voice{voices.length !== 1 ? 's' : ''} |
              지원 언어: {supportedLanguages.map(l => LANGUAGE_NAMES[l] || l).join(', ')}
            </span>
            <span className="text-green-500">
              Zonos TTS - 음성 복제 지원
            </span>
          </div>
        </div>

        {/* Hidden audio element for preview */}
        <audio ref={audioRef} className="hidden" />
      </div>
    </div>
  );
}

export default VoiceManagement;
