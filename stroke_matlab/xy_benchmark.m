test=csvread('..\test.csv',1,0);
ysall=[];
xsall=[];
for signatureID=606:1081
    signatureID
    signatureLength=sum(test(:,2)==signatureID);
    filename=int2str(signatureID);
    while(length(filename)<4)
        filename=['0',filename];
    end
    image=imread(['..\images\',filename,'.jpg']);
    binary=IM2BW(image,0.9);
    zhang=Skeleton(~binary);
    for x=1:size(zhang,2)
        for y=1:size(zhang,1)
            if(zhang(y,x)==1)
                xs=[xs x];
                ys=[ys y];
            end
        end
    end
	%resampling the signals such that the new length will be signatureLength
    xs=spline(1:length(xs),xs,1/signatureLength:(length(xs))/signatureLength:length(xs));
    ys=spline(1:length(ys),ys,1/signatureLength:(length(ys))/signatureLength:length(ys));
	%normalizing the signals into (0,1)
    if max(xs)~=min(xs)
        xs=(xs-min(xs))/(max(xs)-min(xs));
    else
        xs=zeros(length(xs),1);
    end
    if max(ys)~=min(ys)
        ys=(ys-min(ys))/(max(ys)-min(ys));
    else
        ys=zeros(length(ys),1);
    end    
    xsall=[xsall,xs];    
    ysall=[ysall,ys];
    length(xsall)
    length(test(test(:,2)<=signatureID))
end
test(:,6)=xsall';
test(:,7)=ysall';
csvwrite_with_headers('xy_benchmark.csv',test,{'prediction_id','signature_id','writer_id','occurrence_id','time','x','y'});