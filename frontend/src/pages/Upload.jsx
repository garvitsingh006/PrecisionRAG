import FileUploader from '../components/FileUploader'

export default function Upload() {
  return (
    <main className="max-w-5xl mx-auto px-6 py-20 space-y-8">
      <div className="max-w-2xl space-y-2">
        <h1 className="font-display font-semibold text-4xl text-near-white tracking-[-0.8px]">Upload Document</h1>
        <p className="text-muted text-base">
          Upload a PDF — it gets stored in Cloudinary and downloaded to the backend.
          The FAISS index rebuilds automatically on the next query.
        </p>
      </div>
      <div className="max-w-2xl">
        <FileUploader />
      </div>
    </main>
  )
}
