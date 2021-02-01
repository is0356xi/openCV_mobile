package com.example.opencv_mobile

//import io.reactivex.Observable
import rx.Observable
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.PartMap
import okhttp3.RequestBody

interface ApiInterface{
    @JvmSuppressWildcards
    @Multipart
    @POST("/api/post")
    fun ImgPostEntry(
        @PartMap params: Map<String, RequestBody>): Observable<PostResponse>
}
